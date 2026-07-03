import logging
from .utils.data import AdapTestDataset
from tensorboardX import SummaryWriter


class AdapTestDriver(object):
    """自适应测试主循环:选题 -> 记录作答 -> 更新能力估计 -> 评估,重复 test_length 步"""

    @staticmethod
    def run(model, strategy, adaptest_data,
            test_length, log_dir):
        """
        Args:
            model: AbstractModel,认知诊断模型(须已预加载题目参数)
            strategy: AbstractStrategy,选题策略
            adaptest_data: AdapTestDataset,测试学生数据
            test_length: int,考试长度(选题步数)
            log_dir: str,tensorboard 日志目录

        Returns:
            all_results: list of dict,每步的评估指标 [{'auc':…, 'cov':…}, …](含第 0 步)
        """
        writer = SummaryWriter(log_dir)
        all_results = []

        logging.info(f'start adaptive testing with {strategy.name} strategy')

        logging.info(f'Iteration 0')
        # 第 0 步:一题未测时的基线指标(theta 还是随机初始化)
        results = model.adaptest_evaluate(adaptest_data)
        all_results.append(results)
        for name, value in results.items():
            logging.info(f'{name}:{value}')
            writer.add_scalars(name, {strategy.name: value}, 0)

        for it in range(1, test_length + 1):
            logging.info(f'Iteration {it}')
            # 按策略为每个学生选一道题
            selected_questions = strategy.adaptest_select(model, adaptest_data)
            for student, question in selected_questions.items():
                adaptest_data.apply_selection(student, question)
            # 用新作答更新学生能力估计
            model.adaptest_update(adaptest_data)
            # 评估当前估计的 AUC 和知识点覆盖率
            results = model.adaptest_evaluate(adaptest_data)
            all_results.append(results)
            for name, value in results.items():
                logging.info(f'{name}:{value}')
                writer.add_scalars(name, {strategy.name: value}, it)

        return all_results
