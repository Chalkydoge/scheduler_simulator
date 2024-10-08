# config_loader_test.py
import unittest
from config_loader import ConfigLoader  # 假设ConfigLoader类在your_module中
from scheduling_strategies import RandomSchedulingStrategy


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        # 设置文件路径为 /data/setup1.yaml
        self.config_path = './data/setup0.yaml'

        # 使用ConfigLoader加载实际的配置文件
        self.config_loader = ConfigLoader(self.config_path)

    def test_load_nodes(self):
        nodes = self.config_loader.create_nodes()

        # 断言节点数量
        self.assertEqual(len(nodes), 2)

        # 验证每个节点的属性
        self.assertEqual(nodes[0].name, "Node_1")
        self.assertEqual(nodes[0].cpu_capacity, 16)
        self.assertEqual(nodes[0].mem_capacity, 32000)
        self.assertEqual(nodes[0].band_capacity, 1000)

    def test_load_rtt_table(self):
        rtt_table = self.config_loader.create_rtt_table()

        # 断言RTT表正确解析
        self.assertEqual(rtt_table["Node_1"]["Node_2"], 10)
        self.assertEqual(rtt_table["Node_2"]["Node_1"], 10)

    def test_load_pods(self):
        pods = self.config_loader.create_pods()

        # 断言Pod数量
        self.assertEqual(len(pods), 2)

        # 验证Pod的属性
        pod_a = pods["Pod_A"]
        self.assertEqual(pod_a.name, "Pod_A")
        self.assertEqual(pod_a.cpu_resource, 2)
        self.assertEqual(pod_a.mem_resource, 2048)
        self.assertEqual(pod_a.band_resource, 200)

    def test_load_dependencies(self):
        pods = self.config_loader.create_pods()
        dependencies = self.config_loader.create_dependencies(pods)

        # 验证依赖关系
        self.assertIn(pods["Pod_A"], dependencies[pods["Pod_B"]])

    def test_select_strategy(self):
        # 断言调度策略
        strategy = self.config_loader.select_strategy()
        self.assertIsInstance(strategy, RandomSchedulingStrategy)

    def test_create_workload(self):
        pods = self.config_loader.create_pods()
        dependencies = self.config_loader.create_dependencies(pods)
        workload = self.config_loader.create_workload(pods, dependencies)

        # 验证workload属性
        self.assertEqual(workload.num_pods, 2)
        self.assertIn(pods["Pod_A"], workload.pods)
        self.assertIn(pods["Pod_B"], workload.pods)


if __name__ == '__main__':
    unittest.main()
