from abc import ABC, abstractmethod

from gradient_boost import GradientBoostModel, calculate_degradation
from models import Pod, Node, PodAggregator
import random


class SchedulingStrategy(ABC):
    @abstractmethod
    def select_node(self, pod, candidate_nodes, **kwargs) -> Node:
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
        """
            延迟感知调度策略，用于选择最优的节点调度Pod。根据网络延迟和处理延迟选择最优节点。
            Args:
                pod (Pod): 当前待调度的Pod。
                candidate_nodes (list[Node]): 可选的节点列表，用于调度Pod。
                **kwargs: 其他附加参数，包括'rtt'（RTT表），以及目前node上面已经放了的pod数量
            Returns:
                Node: 选择的调度节点。
        """
        alpha = 0.75
        bias = 20
        delays = {}
        # 打印附加参数中的信息，可能包含Pod和RTT
        # for key, value in kwargs.items():
        #     print(f'{key}: {value}')

        # 从kwargs中提取当前的Pod和RTT表
        rtt = kwargs.get('rtt')
        prev_pod_node = kwargs.get('prev_pod_node')  # 表示前一个安放pod的节点，由于跨节点传输会带来额外的延迟
        zeta = kwargs.get('node_workload')  # zeta 表示 node_i 已经被调度上去的pod 的个数

        def calculate_fv_uk(p: Pod, n: Node):
            # Amount of data 但是这个时候data的数量其实是未知的
            transfer_time = (p.data_amount * 1024 * 8) / n.band_capacity
            R = (1.0 / (p.cpu_resource / n.cpu_had))
            """
            S: 数据大小=1Gb, I=20(instruction number for every byte data),
            F=2.5Ghz(CPU freq), C=2
            S*I*C=1Gb*20*2= 40G
            SIC/F = 40/2.5 = 16s
            40x/2.5=16x
            """
            calculate_time = 16 * p.data_amount * R
            return transfer_time + calculate_time

        def calculate_delay_processing(p: Pod, n: Node):
            """
            :param u: 用户的工作
            :param p:  用户工作的pod
            :param n:  当前考虑的node
            :return: 数据处理的延迟，与node位置无关，仅与node自身属性有关
            """
            dp = p.setup_time + calculate_fv_uk(p, n)
            return dp

        def calculate_delay_network(pod_node: str, cur_node: str):
            """
            :param u: 用户的工作
            :param pod_node: 用户的pod所在的node 问题! pod在这里还没有被调度无法得知所在节点的信息!
            :param cur_node: 当前考虑的node
            :return: 网络处理的延迟
                todo 网络处理延迟还会包括拉取image带来的带宽占用 而原文没有考虑这一点
            """
            dn = 0.0
            actual_val = rtt[pod_node].get(cur_node)
            dn += actual_val  # 固定的网络延迟
            return dn

        # 计算每个节点的延迟
        for ni in candidate_nodes:
            cur_node_name = ni.name
            # 计算传播延迟和网络延迟
            delay_processing = calculate_delay_processing(pod, ni)
            delay_networking = 0 if prev_pod_node is None else calculate_delay_network(prev_pod_node, cur_node_name)

            # 计算综合延迟 a_i,j
            score_pod_j_node_i = alpha * delay_processing + (1 - alpha) * delay_networking

            # 如果 ζi > 0 ，添加 bias
            if zeta.get(ni.name) is not None:
                score_pod_j_node_i += bias

            # 保存延迟结果
            delays[ni] = score_pod_j_node_i

        # 选择最小延迟的节点
        n_ref = min(delays, key=delays.get)
        node_selected = [n_ref]

        # 计算 n_ref 的平均延迟
        a_ref = delays[node_selected[0]]
        lambda_threshold = 0.2 * a_ref  # is allowed for [a_ref, 1.2 * a_ref]
        # 检查其他节点的延迟是否在阈值范围内
        for ni in candidate_nodes:
            if ni != n_ref and delays[ni] <= a_ref + lambda_threshold:
                node_selected.append(ni)

        # 如果多个节点在选集 N_SEL 中，选择具有最大 min(rCPU, rMEM) 的节点
        if len(node_selected) > 1:
            return max(node_selected, key=lambda node_sel: max((node_sel.cpu_capacity, node_sel.mem_capacity)))

        # 如果只有一个节点，直接返回 n_ref
        return n_ref


class InterferenceAwareSchedulingStrategy(SchedulingStrategy):
    def select_node(self, pod, candidate_nodes, **kwargs):
        """
        :param pod:
        :param candidate_nodes:
        :param kwargs:
        :return:

        contention vector is calculated as: V(CNF)
        CNF 是当前的cnf状态
        然后我们需要通过regression model知道的是 性能的下降程度P

        然后分部分依次累加数据处理的延迟，网络处理的延迟以及 性能下降的延迟
        最后求出最小化的 (暴力求解)
        """
        alpha = 0.33
        beta = 0.33
        delays = {}

        # 打印附加参数中的信息，可能包含Pod和RTT
        # for key, value in kwargs.items():
        #     print(f'{key}: {value}')

        # 从kwargs中提取当前的Pod和RTT表
        # rtt可能是会变化的
        rtt = kwargs.get('rtt')
        prev_pod_node = kwargs.get('prev_pod_node')  # 表示前一个安放pod的节点，由于跨节点传输会带来额外的延迟

        def calculate_fv_uk(p: Pod, n: Node):
            # Amount of data 但是这个时候data的数量其实是未知的
            transfer_time = (p.data_amount * 1024.0 * 8) / n.band_capacity  # 需要处理的数据 / 带宽(M)
            R = (1.0 / (p.cpu_resource / n.cpu_had))
            calculate_time = 16 * p.data_amount * R
            tot = transfer_time + calculate_time
            return tot

        def calculate_data_processing(p: Pod, n: Node):
            """
            :param u: 用户的工作
            :param p:  用户工作的pod
            :param n:  当前考虑的node
            :return: 数据处理的延迟，与node位置无关，仅与node自身属性有关
            """
            dp = calculate_fv_uk(p, n)
            return dp

        def calculate_setup_time(p: Pod):
            ds = p.setup_time
            return ds

        def calculate_delay_network(pod_node: str, cur_node: str):
            """
            :param u: 用户的工作
            :param pod_node: 用户的pod所在的node 问题! pod在这里还没有被调度无法得知所在节点的信息!
            :param cur_node: 当前考虑的node
            :return: 网络处理的延迟
                todo 网络处理延迟还会包括拉取image带来的带宽占用 而原文没有考虑这一点
            """
            network_latency = rtt[pod_node].get(cur_node)  # 固定的网络延迟
            return network_latency

        def calculate_interference(p: Pod, expected_node: Node):
            # introduce the interference model to solve this
            gb_model = GradientBoostModel()
            gb_model.load_model(gb_model.model_transfer_path, gb_model.model_bitrate_path)
            interference_pods = expected_node.pods
            aggregator = PodAggregator(interference_pods)

            input_x = aggregator.aggregate()
            if all(x == 0.0 for x in input_x):
                return 0  # 因为什么都没有发生
            else:
                y_pred = gb_model.model_transfer.predict(input_x.reshape(1, -1))
                base = calculate_degradation(y_pred)
                # print(base)
                base_latency = calculate_fv_uk(p, expected_node)
                addition_latency = base_latency * (1.0 / base - 1.0)  # 因为干扰 导致处理数据能力下降
                return addition_latency

        # 计算每个节点的延迟
        for ni in candidate_nodes:
            cur_node_name = ni.name
            # print(cur_node_name)
            # 计算传播延迟和网络延迟
            dp = calculate_data_processing(pod, ni)
            dn = 0 if prev_pod_node is None else calculate_delay_network(prev_pod_node, cur_node_name)
            dr = float(calculate_interference(pod, ni))
            ds = calculate_setup_time(pod)
            # print("数据处理延迟 = {}s, 网络传输延迟 = {}s, 干扰延迟 = {}s".format(dp, dn, dr))
            # 计算综合延迟 a_i,j
            score_pod_j_node_i = ds + alpha * dp + beta * dn + (1.0 - alpha - beta) * dr

            # 保存延迟结果
            delays[ni] = score_pod_j_node_i

        # 选择最小延迟的节点
        n_ref = min(delays, key=delays.get)
        node_selected = [n_ref]

        # 计算 n_ref 的平均延迟
        a_ref = delays[node_selected[0]]
        lambda_threshold = 0.1 * a_ref  # is allowed for [a_ref, 1.2 * a_ref]
        # 检查其他节点的延迟是否在阈值范围内
        for ni in candidate_nodes:
            if ni != n_ref and delays.get(ni) is not None and delays[ni] <= a_ref + lambda_threshold:
                node_selected.append(ni)

        # 如果多个节点在选集 N_SEL 中，选择具有最大 min(rCPU, rMEM) 的节点
        if len(node_selected) > 1:
            return max(node_selected, key=lambda node_sel: node_sel.cpu_capacity)

        # 如果只有一个节点，直接返回 n_ref
        return n_ref


class NetMarksSchedulingStrategy(SchedulingStrategy):
    def select_node(self, pod, candidate_nodes, **kwargs):
        delays = {}
        # 从kwargs中提取当前的Pod和RTT表
        rtt = kwargs.get('rtt')
        prev_pod_node = kwargs.get('prev_pod_node')  # 表示前一个安放pod的节点，由于跨节点传输会带来额外的延迟

        # 计算每个节点的延迟
        for ni in candidate_nodes:
            cur_node_name = ni.name
            # 仅考虑网络延迟
            if prev_pod_node is not None:
                delay_networking = rtt[prev_pod_node].get(cur_node_name)
            else:
                delay_networking = 0
            # 计算综合延迟 a_i,j
            score_pod_j_node_i = delay_networking

            # 保存延迟结果
            delays[ni] = score_pod_j_node_i

        # 选择最小延迟的节点
        n_ref = min(delays, key=delays.get)
        return n_ref


class IPlaceSchedulingStrategy(SchedulingStrategy):
    def select_node(self, pod, candidate_nodes, **kwargs):
        delays = {}
        # 从kwargs中提取当前的Pod和RTT表
        # rtt = kwargs.get('rtt')
        # prev_pod_node = kwargs.get('prev_pod_node')  # 表示前一个安放pod的节点，由于跨节点传输会带来额外的延迟
        ratio = 100.0

        def calculate_interference(expected_node: Node):
            # introduce the interference model to solve this
            gb_model = GradientBoostModel()
            gb_model.load_model(gb_model.model_transfer_path, gb_model.model_bitrate_path)
            interference_pods = expected_node.pods
            aggregator = PodAggregator(interference_pods)

            input_x = aggregator.aggregate()
            if all(x == 0.0 for x in input_x):
                return 0  # 因为什么都没有发生
            else:
                y_pred = gb_model.model_transfer.predict(input_x.reshape(1, -1))
                base = calculate_degradation(y_pred)
                return 1.0 / base

        # 计算每个节点的延迟
        for ni in candidate_nodes:
            # 计算干扰影响和启动延迟
            dr = float(calculate_interference(ni))
            ds = pod.setup_time

            # 计算综合延迟 a_i,j
            score_pod_j_node_i = ds + ratio * dr
            # 保存延迟结果
            delays[ni] = score_pod_j_node_i

        # 选择干扰min的部署方式
        n_ref = min(delays, key=delays.get)
        available_nodes = []
        threshold = 0.2 * delays[n_ref]
        for ni in delays.keys():
            if abs(delays[ni] - delays[n_ref]) <= threshold:
                available_nodes.append(ni)

        # 然后排序 挑选一个尽可能之前部署过的
        available_nodes.sort(key=lambda x: delays[x])
        for ni in available_nodes:
            if len(ni.pods) > 0:
                n_ref = ni
                break

        return n_ref
