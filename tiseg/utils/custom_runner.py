from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS


@RUNNERS.register_module()
class CustomRunner(EpochBasedRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)

            visual = outputs['extra_vars'].pop('visual')
            img = visual['img'][0]
            sem = visual['sem_gt'][0]
            inst = visual['inst_gt'][0]
            print(img.shape, sem.shape, inst.shape)

            import matplotlib.pyplot as plt
            plt.subplot(221)
            plt.imshow(img)
            plt.subplot(222)
            plt.imshow(sem)
            # plt.subplot(223)
            # plt.imshow(full_dir_ce_loss)
            # plt.subplot(224)
            # plt.imshow(full_dir_ce_loss_dirw)
            plt.savefig('2.png')
            print(outputs)
            exit(0)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"' 'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
