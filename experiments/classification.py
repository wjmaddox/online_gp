import hydra
import random
import gpytorch
import time
import pandas as pd

from upcycle.random.seed import set_all_seeds
from omegaconf import OmegaConf, DictConfig
from gpytorch.settings import *
from online_gp.settings import detach_interp_coeff


def startup(hydra_cfg):
    if hydra_cfg.seed is None:
        seed = random.randint(0, 100000)
        hydra_cfg['seed'] = seed
        set_all_seeds(seed)

    logger = hydra.utils.instantiate(hydra_cfg.logger)
    hydra_cfg = OmegaConf.to_container(hydra_cfg, resolve=True)  # Resolve config interpolations
    hydra_cfg = DictConfig(hydra_cfg)
    logger.write_hydra_yaml(hydra_cfg)

    print(hydra_cfg.pretty())
    print(f"GPU available: {torch.cuda.is_available()}")

    return hydra_cfg, logger


def get_model(config, init_x, init_y, streaming):
    stem = hydra.utils.instantiate(config.stem)
    model_kwargs = dict(stem=stem, init_x=init_x, init_y=init_y)
    model = hydra.utils.instantiate(config.model, **model_kwargs)
    if torch.cuda.is_available():
        model = model.cuda()

    return model


def online_learning(batch_model, online_model, train_x, train_y, test_x, test_y, update_stem,
                    logger, logging_freq):
    online_correct, batch_correct = 0, 0
    logger.add_table('online_metrics')

    for t, (x, y) in enumerate(zip(train_x, train_y)):
        start_clock = time.time()
        with detach_interp_coeff(True):
            online_pred_y = online_model.predict(x)
        stem_loss, gp_loss = online_model.update(x, y, update_stem=update_stem)
        step_time = time.time() - start_clock

        online_correct += online_pred_y.eq(y).sum().float().item()
        with torch.no_grad(), gpytorch.settings.skip_posterior_variances(False):
            batch_pred_y = batch_model.predict(x)
        batch_correct += batch_pred_y.eq(y).sum().float().item()
        regret = batch_correct - online_correct

        if t % logging_freq == (logging_freq - 1):
            test_pred_y = online_model.predict(test_x)
            test_acc = test_pred_y.eq(test_y).float().mean().item()
            print(f'T: {t+1}, test accuracy: {test_acc:0.4f}, regret: {regret}')

            logger.log(dict(
                stem_loss=stem_loss,
                gp_loss=gp_loss,
                batch_correct=batch_correct,
                online_correct=online_correct,
                regret=regret,
                test_acc=test_acc,
                step_time=step_time
            ), step=t + 1, table_name='online_metrics')
            logger.write_csv()


def classification_trial(config):
    config, logger = startup(config)

    datasets = hydra.utils.instantiate(config.dataset)
    train_x, train_y = datasets.train_dataset[:]
    test_x, test_y = datasets.test_dataset[:]
    config.stem.input_dim = config.dataset.input_dim = train_x.size(-1)

    print('==== training model in batch setting ====')
    batch_model = get_model(config, train_x, train_y, streaming=False)
    batch_model.set_lr(config.model.batch_gp_lr, config.model.batch_stem_lr)
    batch_metrics = batch_model.fit(train_x, train_y, config.num_batch_epochs, datasets.test_dataset)
    logger.add_table('batch_metrics', batch_metrics)
    logger.write_csv()
    batch_df = pd.DataFrame(logger.data['batch_metrics'], index=None)
    print(batch_df.tail(5).to_markdown())

    print('==== training model in online setting ====')
    num_init_obs = int(config.model.init_ratio * train_x.size(0))
    init_x, train_x = train_x[:num_init_obs], train_x[num_init_obs:]
    init_y, train_y = train_y[:num_init_obs], train_y[num_init_obs:]
    online_model = get_model(config, init_x, init_y, streaming=True)

    if config.pretrain:
        print('==== pretraining ====')
        online_model.set_lr(config.model.batch_gp_lr, config.model.batch_stem_lr)
        pretrain_metrics = online_model.fit(init_x, init_y, config.num_batch_epochs, datasets.test_dataset)
        logger.add_table('pretrain_metrics', pretrain_metrics)
        logger.write_csv()
        pretrain_df = pd.DataFrame(logger.data['pretrain_metrics'], index=None)
        print(pretrain_df.tail(5).to_markdown())

    online_model.set_lr(config.model.online_gp_lr, config.model.online_stem_lr)
    online_learning(batch_model, online_model, train_x, train_y, test_x, test_y,
                    config.update_stem, logger, config.logging_freq)
    online_df = pd.DataFrame(logger.data['online_metrics'], index=None)
    print(online_df.tail(5).to_markdown())


@hydra.main(config_path='../config/classification.yaml')
def main(config):
    with max_root_decomposition_size(config.gpytorch_global_settings.max_root_decomposition_size),\
         max_cholesky_size(config.gpytorch_global_settings.max_cholesky_size),\
         cg_tolerance(config.gpytorch_global_settings.cg_tolerance):
        classification_trial(config)


if __name__ == '__main__':
    main()
