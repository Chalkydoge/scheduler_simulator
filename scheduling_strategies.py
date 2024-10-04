from abc import ABC, abstractmethod
import random

from models import UserWorkload, Pod, Node


class SchedulingStrategy(ABC):
    @abstractmethod
    def select_node(self, pod, candidate_nodes, **kwargs):
        """从候选节点中选择一个节点来调度 Pod"""
        pass


class RandomSchedulingStrategy(SchedulingStrategy):
    def select_node(self, pod, candidate_nodes, **kwargs):
        return random.choice(candidate_nodes)


class LeastResourceSchedulingStrategy(SchedulingStrategy):
    def select_node(self, pod, candidate_nodes, **kwargs):
        return max(candidate_nodes, key=lambda n: (n.cpu_capacity, n.mem_capacity, n.band_capacity))


class DelayAwareSchedulingStrategy(SchedulingStrategy):
    def select_node(self, pod, candidate_nodes, **kwargs):
        alpha = 0.1
        zeta_i = 0
        lambda_threshold = 114514
        bias = 0
        delays = {}
        for key, value in kwargs.items():
            print(f'{key}: {value}')
        # 用户Workload信息 and 用户workload当前关心的pod信息
        uk = kwargs.get('user')
        pj = kwargs.get('pod')

        def calculate_delay_processing(u: UserWorkload, p: Pod, n: Node):
            """
            :param u: 用户的工作
            :param p:  用户工作的pod
            :param n:  当前考虑的node
            :return: 数据处理的延迟，与node位置无关，仅与node自身属性有关
            """
            dp = 0.0
            return dp

        def calculate_delay_network(u: UserWorkload, pod_node: Node, cur_node: Node):
            """
            :param u: 用户的工作
            :param pod_node: 用户的pod所在的node 问题! pod在这里还没有被调度无法得知所在节点的信息!
            :param cur_node: 当前考虑的node
            :return: 网络处理的延迟
            todo 网络处理延迟还会包括拉取image带来的带宽占用 而原文没有考虑这一点
            """
            dn = 0.0
            return dn

        # 计算每个节点的延迟
        for ni in candidate_nodes:
            # 计算传播延迟和网络延迟


            Dp = calculate_delay_processing(uk, pj, ni)
            Dn = calculate_delay_network(uk, pj, ni)

            # 计算综合延迟 a_i,j
            a_i_j = alpha * Dp + (1 - alpha) * Dn

            # 如果 ζi > 0 ，添加 bias
            if zeta_i > 0:
                a_i_j += bias

            # 保存延迟结果
            delays[ni] = a_i_j

        # 选择最小延迟的节点
        n_ref = min(delays, key=delays.get)
        node_selected = [n_ref]

        # 计算 n_ref 的平均延迟
        a_ref = delays[node_selected]

        # 检查其他节点的延迟是否在阈值范围内
        for ni in candidate_nodes:
            if ni != n_ref and delays[ni] <= a_ref + lambda_threshold:
                node_selected.append(ni)

        # 如果多个节点在选集 N_SEL 中，选择具有最大 min(rCPU, rMEM) 的节点
        if len(node_selected) > 1:
            return max(node_selected, key=lambda ni: min(ni.rCPU, ni.rMEM))

        # 如果只有一个节点，直接返回 n_ref
        return n_ref
