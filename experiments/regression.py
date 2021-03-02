import hydra
import random
from omegaconf import OmegaConf, DictConfig
from upcycle.random.seed import set_all_seeds
import time
import pandas as pd
from online_gp.utils.dkl import pretrain_stem
from gpytorch.settings import *
from upcycle import cuda


def startup(hydra_cfg):
    if hydra_cfg.seed is None:
        seed = random.randint(0, 100000)
        hydra_cfg['seed'] = seed
        set_all_seeds(seed)

    logger = hydra.utils.instantiate(hydra_cfg.logger)
    hydra_cfg = OmegaConf.to_container(hydra_cfg, resolve=True)  # Resolve config interpolations
    hydra_cfg = DictConfig(hydra_cfg)
    logger.write_hydra_yaml(hydra_cfg)

    if hydra_cfg.dtype == 'float32':
        torch.set_default_dtype(torch.float32)
    elif hydra_cfg.dtype == 'float64':
        torch.set_default_dtype(torch.float64)

    print(hydra_cfg.pretty())
    print(f"GPU available: {torch.cuda.is_available()}")

    return hydra_cfg, logger


def get_model(config, init_x, init_y, streaming):
    stem = hydra.utils.instantiate(config.stem)
    model_kwargs = dict(stem=stem, init_x=init_x, init_y=init_y)
    model = hydra.utils.instantiate(config.model, **model_kwargs)
    return cuda.try_cuda(model)


def online_regression(batch_model, online_model, train_x, train_y, test_x, test_y,
                      update_stem, batch_size, logger, logging_freq):
    online_rmse = online_nll = 0
    batch_rmse = batch_nll = 0
    logger.add_table('online_metrics')
    num_chunks = train_x.size(-2) // batch_size

    for t, (x, y) in enumerate(zip(train_x.chunk(num_chunks), train_y.chunk(num_chunks))):
        start_clock = time.time()
        from online_gp.settings import detach_interp_coeff
        with detach_interp_coeff(True):
            o_rmse, o_nll = online_model.evaluate(x, y)
        stem_loss, gp_loss = online_model.update(x, y, update_stem=update_stem)
        step_time = time.time() - start_clock

        with torch.no_grad():
            b_rmse, b_nll = batch_model.evaluate(x, y)
        online_rmse += o_rmse
        online_nll += o_nll
        batch_rmse += b_rmse
        batch_nll += b_nll

        regret = online_rmse - batch_rmse
        num_steps = (t + 1) * batch_size
        if t % logging_freq == (logging_freq - 1):
            rmse, nll = online_model.evaluate(test_x, test_y)
            print(f'T: {t+1}, test RMSE: {rmse:0.4f}, test NLL: {nll:0.4f}')
            logger.log(dict(
                stem_loss=stem_loss,
                gp_loss=gp_loss,
                batch_rmse=batch_rmse,
                batch_nll=batch_nll,
                online_rmse=online_rmse,
                online_nll=online_nll,
                regret=regret,
                test_rmse=rmse,
                test_nll=nll,
                noise=online_model.noise.mean().item(),
                step_time=step_time
            ), step=num_steps, table_name='online_metrics')
            logger.write_csv()


def regression_trial(config):
    config, logger = startup(config)

    datasets = hydra.utils.instantiate(config.dataset)
    train_x, train_y = datasets.train_dataset[:]
    test_x, test_y = datasets.test_dataset[:]
    config.stem.input_dim = config.dataset.input_dim = train_x.size(-1)


    batch_model = get_model(config, train_x, train_y, streaming=False)

    if config.pretrain_stem.enabled:
        print('==== pretraining stem ====')
        loss_fn = torch.nn.MSELoss()
        batch_pretrain_stem_metrics = pretrain_stem(batch_model.stem, train_x, train_y, loss_fn,
                                                    **config.pretrain_stem)
        logger.add_table('batch_pretrain_stem_metrics', batch_pretrain_stem_metrics)
        logger.write_csv()
        pretrain_df = pd.DataFrame(logger.data['batch_pretrain_stem_metrics'])
        print(pretrain_df.tail(5).to_markdown())

    print('==== training GP in batch setting ====')
    batch_model.set_lr(gp_lr=config.dataset.base_lr, stem_lr=config.dataset.base_lr / 10)
    batch_metrics = batch_model.fit(train_x, train_y, config.num_batch_epochs, datasets.test_dataset)
    logger.add_table('batch_metrics', batch_metrics)
    logger.write_csv()
    batch_df = pd.DataFrame(logger.data['batch_metrics'], index=None)
    print(batch_df.tail(5).to_markdown())

    num_init_obs = int(config.model.init_ratio * train_x.size(0))
    init_x, train_x = train_x[:num_init_obs], train_x[num_init_obs:]
    init_y, train_y = train_y[:num_init_obs], train_y[num_init_obs:]
    print(f'==== training model in online setting, N: {train_x.size(0)} ====')
    online_model = get_model(config, init_x, init_y, streaming=True)

    if config.pretrain_stem.enabled:
        print('==== pretraining stem ====')
        loss_fn = torch.nn.MSELoss()
        online_pretrain_stem_metrics = pretrain_stem(online_model.stem, init_x, init_y, loss_fn,
                                                     **config.pretrain_stem)
        logger.add_table('online_pretrain_stem_metrics', online_pretrain_stem_metrics)
        logger.write_csv()
        pretrain_df = pd.DataFrame(logger.data['online_pretrain_stem_metrics'])
        print(pretrain_df.tail(5).to_markdown())

    if config.pretrain:
        print('==== pretraining gp ====')
        online_model.set_lr(gp_lr=config.dataset.base_lr, stem_lr=config.dataset.base_lr / 10)
        pretrain_metrics = online_model.fit(init_x, init_y, config.num_batch_epochs, datasets.test_dataset)
        logger.add_table('pretrain_metrics', pretrain_metrics)
        logger.write_csv()
        pretrain_df = pd.DataFrame(logger.data['pretrain_metrics'])
        print(pretrain_df.tail(5).to_markdown())

    online_model.set_lr(gp_lr=config.dataset.base_lr / 10, stem_lr=config.dataset.base_lr / 100)
    online_regression(batch_model, online_model, train_x, train_y, test_x, test_y,
                      config.update_stem, config.batch_size, logger, config.logging_freq)
    online_df = pd.DataFrame(logger.data['online_metrics'], index=None)
    print(online_df.tail(5).to_markdown())


@hydra.main(config_path='../config/regression.yaml')
def main(config):
    with max_root_decomposition_size(config.gpytorch_global_settings.max_root_decomposition_size),\
         max_cholesky_size(config.gpytorch_global_settings.max_cholesky_size),\
         cg_tolerance(config.gpytorch_global_settings.cg_tolerance):
        regression_trial(config)


if __name__ == '__main__':
    main()
