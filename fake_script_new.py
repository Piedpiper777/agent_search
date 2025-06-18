"""
Agent Search System - 多目标动态规划+强化学习+LLM智能 完整伪代码
结合DP的最优性、RL的学习能力、LLM的智能推理
"""
#RL能不能集成到LLM中呢

import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

def AgentAction(value: str):
    """模拟AgentAction枚举"""
    return value

# ===== 1. 核心数据结构 =====
@dataclass
class QuantitativeState:
    """量化状态表示，记录量化的Agent状态"""
    stage: int                      # 决策阶段 t=0,1,2,...（状态转移方程的阶段？）
    facts_completeness: float       # 事实完整度 [0,1]
    query_complexity: float         # 查询复杂度 [0,1]
    search_exhaustion: float        # 搜索穷尽度 [0,1]
    time_elapsed: float             # 时间消耗比例 [0,1]
    confidence_level: float         # 当前置信度 [0,1]
    
    def to_key(self) -> str:
        """转换为状态键（用于记忆化）"""
        return f"{self.stage}_{self.facts_completeness:.1f}_{self.query_complexity:.1f}_{self.time_elapsed:.1f}_{self.search_exhaustion:.1f}_{self.confidence_level:.1f}"
    
    def to_vector(self) -> np.ndarray:
        """转换为向量形式"""
        return np.array([
            self.facts_completeness,
            self.query_complexity, 
            self.search_exhaustion,
            self.time_elapsed,
            self.confidence_level
        ])

@dataclass
class ParetoSolution:
    """帕累托最优解"""
    action: AgentAction
    state: QuantitativeState
    objectives: np.ndarray          # [准确性, 效率, 完整性, 成本]，目标向量
    policy_path: List[AgentAction]  # 从当前状态到终点的动作序列
    expected_reward: float          # 期望总奖励
    confidence: float               # 解的置信度

# ===== 2. 多目标动态规划求解器 =====
class MultiObjectiveDynamicProgramming:
    """多目标动态规划求解器"""
    
    def __init__(self):
        # 帕累托最优解记忆表: state_key -> List[ParetoSolution]，用来存储已计算的帕累托解
        self.pareto_memo = {}
        
        # 状态转移函数
        self.state_transitions = StateTransitionModel()
        
        # 多目标奖励函数
        self.reward_functions = MultiObjectiveRewards()
        
        # 动作可行性检查器
        self.feasibility_checker = ActionFeasibilityChecker()
        
        # 算法参数
        self.max_stages = 10 # 最大决策阶段
        self.gamma = 0.9  # 折扣因子
        
    def solve(self, initial_state: QuantitativeState) -> List[ParetoSolution]:
        """主求解函数"""
        print(f"开始多目标动态规划求解，初始状态: {initial_state}")
        
        # 清空记忆表（可选）
        if len(self.pareto_memo) > 1000:  # 防止内存溢出
            self.pareto_memo.clear()
        
        # 递归求解
        pareto_solutions = self._solve_recursive(initial_state, self.max_stages) #递归包含在_solve_recursive中
        
        print(f"找到 {len(pareto_solutions)} 个帕累托最优解")
        return pareto_solutions
    
    def _solve_recursive(self, state: QuantitativeState, stages_remaining: int) -> List[ParetoSolution]:
        """递归求解单个状态"""
        state_key = state.to_key() # 用于记忆化的状态键
        
        # Step 1: 记忆化检查
        if state_key in self.pareto_memo:
            return self.pareto_memo[state_key] # 如果已经计算过这个状态，直接返回记忆中的解
        
        # Step 2: 边界条件检查
        if stages_remaining == 0 or self._is_terminal_state(state):
            terminal_solutions = self._get_terminal_solutions(state)
            self.pareto_memo[state_key] = terminal_solutions
            return terminal_solutions  # 如果没有剩余阶段或是终止状态，返回终止解
        
        # Step 3: 获取可行动作
        feasible_actions = self._get_feasible_actions(state)
        if not feasible_actions:
            # 没有可行动作，只能终止
            return self._get_terminal_solutions(state)
        
        # Step 4: 遍历所有可行动作，收集候选解
        all_candidate_solutions = []
        
        for action in feasible_actions:
            # 计算立即奖励向量
            immediate_rewards = self.reward_functions.calculate_immediate_reward(state, action)
            
            # 获取可能的下一状态分布
            next_state_distribution = self.state_transitions.get_next_states(state, action)
            
            # 对每个可能的下一状态
            for next_state, transition_prob in next_state_distribution:
                # 递归求解下一状态的帕累托最优解
                future_solutions = self._solve_recursive(next_state, stages_remaining - 1)
                
                # 组合当前奖励和未来最优解
                for future_solution in future_solutions:
                    combined_objectives = self._combine_objectives(
                        immediate_rewards, 
                        future_solution.objectives, 
                        transition_prob
                    )
                    
                    # 创建候选帕累托解
                    candidate_solution = ParetoSolution(
                        action=action,
                        state=state,
                        objectives=combined_objectives,
                        policy_path=[action] + future_solution.policy_path,
                        expected_reward=np.sum(combined_objectives),
                        confidence=transition_prob * future_solution.confidence
                    )
                    
                    all_candidate_solutions.append(candidate_solution)
        
        # Step 5: 筛选帕累托最优解
        pareto_optimal_solutions = self._filter_pareto_optimal(all_candidate_solutions)
        
        # Step 6: 记忆化存储
        self.pareto_memo[state_key] = pareto_optimal_solutions
        
        return pareto_optimal_solutions
    
    def _get_feasible_actions(self, state: QuantitativeState) -> List[AgentAction]:
        """获取可行动作集合"""
        all_actions = [
            AgentAction.DECOMPOSE,
            AgentAction.SEARCH_KG, 
            AgentAction.SEARCH_VECTOR,
            AgentAction.GENERATE,
            AgentAction.REFINE
        ]
        
        feasible = []
        for action in all_actions:
            if self.feasibility_checker.is_feasible(state, action):
                feasible.append(action)
        
        return feasible
    
    def _is_terminal_state(self, state: QuantitativeState) -> bool:
        """判断是否为终止状态"""
        # 终止条件：
        # 1. 事实完整度足够高且置信度高
        # 2. 时间用完
        # 3. 已经生成了答案
        return (
            (state.facts_completeness >= 0.8 and state.confidence_level >= 0.7) or
            state.time_elapsed >= 0.95 or
            state.stage >= self.max_stages
        )
    
    def _get_terminal_solutions(self, state: QuantitativeState) -> List[ParetoSolution]:
        """获取终止状态的解"""
        # 终止状态只有一个动作：生成答案
        terminal_reward = self.reward_functions.calculate_terminal_reward(state)
        
        return [ParetoSolution(
            action=AgentAction.GENERATE,
            state=state,
            objectives=terminal_reward,
            policy_path=[AgentAction.GENERATE],
            expected_reward=np.sum(terminal_reward),
            confidence=state.confidence_level
        )]
    
    def _combine_objectives(self, immediate: np.ndarray, future: np.ndarray, prob: float) -> np.ndarray:
        """组合当前奖励和未来奖励"""
        return immediate + self.gamma * prob * future
    
    def _filter_pareto_optimal(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """筛选帕累托最优解"""
        if not solutions:
            return []
        
        pareto_optimal = []
        
        for i, sol1 in enumerate(solutions):
            is_dominated = False
            
            for j, sol2 in enumerate(solutions):
                if i != j and self._dominates(sol2.objectives, sol1.objectives):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(sol1)
        
        return pareto_optimal
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """判断obj1是否帕累托支配obj2"""
        # obj1支配obj2当且仅当：
        # 1. 在所有目标上 obj1 >= obj2
        # 2. 至少在一个目标上 obj1 > obj2
        all_geq = np.all(obj1 >= obj2)
        any_greater = np.any(obj1 > obj2)
        return all_geq and any_greater

# ===== 3. 多目标奖励函数 =====
class MultiObjectiveRewards:
    """多目标奖励函数"""
    
    def __init__(self):
        self.objective_weights = {
            'accuracy': 0.4,
            'efficiency': 0.25,
            'completeness': 0.25, 
            'cost': 0.1
        }
    
    def calculate_immediate_reward(self, state: QuantitativeState, action: AgentAction) -> np.ndarray:
        """计算立即奖励向量 [准确性, 效率, 完整性, 成本]"""
        
        if action == AgentAction.DECOMPOSE:
            return np.array([
                0.1 * state.query_complexity,    # 准确性：复杂查询分解后更准确
                0.8 - 0.1 * state.time_elapsed,  # 效率：分解快但消耗时间
                0.2,                              # 完整性：分解本身不直接增加完整性
                -0.1                              # 成本：轻量级操作
            ])
            
        elif action == AgentAction.SEARCH_KG:
            return np.array([
                0.7 + 0.1 * (1 - state.search_exhaustion),  # 准确性：KG搜索准确性高
                0.5 - 0.2 * state.time_elapsed,             # 效率：中等效率，时间影响大
                0.6 * (1 - state.search_exhaustion),        # 完整性：取决于已搜索程度
                -0.2                                         # 成本：中等成本
            ])
            
        elif action == AgentAction.SEARCH_VECTOR:
            return np.array([
                0.6 + 0.05 * (1 - state.search_exhaustion), # 准确性：向量搜索准确性中等
                0.7 - 0.1 * state.time_elapsed,             # 效率：较高效率
                0.5 * (1 - state.search_exhaustion),        # 完整性：中等完整性提升
                -0.15                                        # 成本：较低成本
            ])
            
        elif action == AgentAction.GENERATE:
            # 生成答案的质量很大程度依赖于已收集的事实
            completeness_bonus = min(1.0, state.facts_completeness * 1.2)
            return np.array([
                0.8 * completeness_bonus,        # 准确性：依赖事实完整性
                0.9,                             # 效率：生成很快
                0.0,                             # 完整性：不再增加
                -0.3 * completeness_bonus        # 成本：高质量生成成本高
            ])
            
        elif action == AgentAction.REFINE:
            return np.array([
                0.3 * state.confidence_level,    # 准确性：基于当前置信度
                0.4,                             # 效率：精化需要时间
                0.2,                             # 完整性：略微提升
                -0.2                             # 成本：中等成本
            ])
        
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    def calculate_terminal_reward(self, state: QuantitativeState) -> np.ndarray:
        """计算终止状态奖励"""
        # 终止奖励基于最终状态质量
        return np.array([
            state.confidence_level,              # 准确性：最终置信度
            1.0 - state.time_elapsed,           # 效率：剩余时间
            state.facts_completeness,           # 完整性：事实完整度
            -state.time_elapsed                  # 成本：负的时间消耗
        ])

# ===== 4. 状态转移模型 =====
class StateTransitionModel:
    """状态转移模型"""
    
    def get_next_states(self, current_state: QuantitativeState, action: AgentAction) -> List[Tuple[QuantitativeState, float]]:
        """获取下一状态分布 [(next_state, probability), ...]"""
        
        if action == AgentAction.DECOMPOSE:
            # 分解动作：确定性转移
            next_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=current_state.facts_completeness,
                query_complexity=max(0.2, current_state.query_complexity - 0.3),  # 降低复杂度
                search_exhaustion=current_state.search_exhaustion,
                time_elapsed=current_state.time_elapsed + 0.1,
                confidence_level=current_state.confidence_level + 0.1
            )
            return [(next_state, 1.0)]
            
        elif action == AgentAction.SEARCH_KG:
            # 搜索动作：随机性转移（搜索结果不确定）
            base_facts_gain = 0.3
            base_time_cost = 0.2
            
            # 高收益情况（60%概率）
            high_gain_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain + 0.2),
                query_complexity=current_state.query_complexity,
                search_exhaustion=min(1.0, current_state.search_exhaustion + 0.3),
                time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost),
                confidence_level=min(1.0, current_state.confidence_level + 0.2)
            )
            
            # 低收益情况（40%概率）
            low_gain_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain - 0.1),
                query_complexity=current_state.query_complexity,
                search_exhaustion=min(1.0, current_state.search_exhaustion + 0.2),
                time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost + 0.1),
                confidence_level=min(1.0, current_state.confidence_level + 0.05)
            )
            
            return [(high_gain_state, 0.6), (low_gain_state, 0.4)]
            
        elif action == AgentAction.SEARCH_VECTOR:
            # 向量搜索：中等随机性
            base_facts_gain = 0.25
            base_time_cost = 0.15
            
            good_result_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain + 0.1),
                query_complexity=current_state.query_complexity,
                search_exhaustion=min(1.0, current_state.search_exhaustion + 0.25),
                time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost),
                confidence_level=min(1.0, current_state.confidence_level + 0.15)
            )
            
            poor_result_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain - 0.05),
                query_complexity=current_state.query_complexity,
                search_exhaustion=min(1.0, current_state.search_exhaustion + 0.2),
                time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost + 0.05),
                confidence_level=min(1.0, current_state.confidence_level + 0.08)
            )
            
            return [(good_result_state, 0.7), (poor_result_state, 0.3)]
        
        elif action == AgentAction.GENERATE:
            # 生成动作：转移到终止状态
            terminal_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=current_state.facts_completeness,
                query_complexity=current_state.query_complexity,
                search_exhaustion=current_state.search_exhaustion,
                time_elapsed=min(1.0, current_state.time_elapsed + 0.1),
                confidence_level=current_state.confidence_level
            )
            return [(terminal_state, 1.0)]
        
        # 默认：状态不变
        return [(current_state, 1.0)]

# ===== 5. 动作可行性检查器 =====
class ActionFeasibilityChecker:
    """动作可行性检查器"""
    
    def __init__(self):
        self.constraints = {
            'max_time': 0.95,
            'min_confidence_for_generation': 0.6,
            'max_search_exhaustion': 0.9,
            'max_stages': 10
        }
    
    def is_feasible(self, state: QuantitativeState, action: AgentAction) -> bool:
        """检查动作是否可行"""
        
        # 通用约束：时间和阶段限制
        if (state.time_elapsed >= self.constraints['max_time'] or 
            state.stage >= self.constraints['max_stages']):
            return action == AgentAction.GENERATE  # 只能生成答案
        
        # 具体动作约束
        if action == AgentAction.DECOMPOSE:
            # 分解约束：查询复杂度必须足够高，且之前没有分解过
            return state.query_complexity > 0.5 and state.stage <= 1
            
        elif action == AgentAction.SEARCH_KG:
            # KG搜索约束：搜索还没穷尽
            return state.search_exhaustion < self.constraints['max_search_exhaustion']
            
        elif action == AgentAction.SEARCH_VECTOR:
            # 向量搜索约束：搜索还没穷尽
            return state.search_exhaustion < self.constraints['max_search_exhaustion']
            
        elif action == AgentAction.GENERATE:
            # 生成约束：置信度足够或时间不够了
            return (state.confidence_level >= self.constraints['min_confidence_for_generation'] or
                   state.time_elapsed >= 0.8)
            
        elif action == AgentAction.REFINE:
            # 精化约束：已经有一些事实了
            return state.facts_completeness > 0.3
        
        return True

# ===== 6. 强化学习组件 =====
class MultiObjectiveQLearning:
    """多目标Q学习"""
    
    def __init__(self):
        # 帕累托Q值集合：(state, action) -> List[q_vector]
        self.pareto_q_sets = defaultdict(list)
        
        # 学习参数
        self.learning_rate = 0.1
        self.epsilon = 0.1  # 探索率
        self.gamma = 0.9    # 折扣因子
        
        # 经验回放缓冲区
        self.experience_buffer = deque(maxsize=10000)
        
        # 权重学习器
        self.weight_learner = WeightLearner()
    
    def update_q_values(self, experience: Tuple[QuantitativeState, AgentAction, np.ndarray, QuantitativeState]):
        """更新Q值"""
        state, action, reward_vector, next_state = experience
        
        # 添加到经验缓冲区
        self.experience_buffer.append(experience)
        
        # 获取下一状态的最大Q值向量集合
        next_max_q_vectors = self._get_max_q_vectors(next_state)
        
        # 为每个可能的未来Q值向量更新当前Q值
        state_action_key = self._state_action_key(state, action)
        
        new_q_vectors = []
        for future_q in next_max_q_vectors:
            # 多目标Q学习更新规则
            target_q = reward_vector + self.gamma * future_q
            
            # 如果是第一次访问这个状态-动作对
            if not self.pareto_q_sets[state_action_key]:
                new_q_vectors.append(target_q)
            else:
                # 更新现有Q值（使用学习率）
                for existing_q in self.pareto_q_sets[state_action_key]:
                    updated_q = existing_q + self.learning_rate * (target_q - existing_q)
                    new_q_vectors.append(updated_q)
        
        # 筛选帕累托最优Q值
        self.pareto_q_sets[state_action_key] = self._filter_pareto_q_vectors(new_q_vectors)
    
    def select_action(self, state: QuantitativeState, available_actions: List[AgentAction]) -> AgentAction:
        """基于多目标Q值选择动作"""
        
        # epsilon-greedy探索
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # 贪婪选择：对每个动作计算期望Q值
        action_scores = {}
        current_weights = self.weight_learner.get_current_weights()
        
        for action in available_actions:
            state_action_key = self._state_action_key(state, action)
            q_vectors = self.pareto_q_sets.get(state_action_key, [np.zeros(4)])
            
            # 计算加权平均Q值
            weighted_q_values = []
            for q_vector in q_vectors:
                weighted_q = np.dot(current_weights, q_vector)
                weighted_q_values.append(weighted_q)
            
            action_scores[action] = max(weighted_q_values) if weighted_q_values else 0.0
        
        # 选择最高分动作
        best_action = max(action_scores, key=action_scores.get)
        return best_action
    
    def _get_max_q_vectors(self, state: QuantitativeState) -> List[np.ndarray]:
        """获取状态的最大Q值向量集合"""
        all_q_vectors = []
        
        # 收集所有可能动作的Q值向量
        for action in AgentAction:
            state_action_key = self._state_action_key(state, action)
            q_vectors = self.pareto_q_sets.get(state_action_key, [np.zeros(4)])
            all_q_vectors.extend(q_vectors)
        
        # 返回帕累托最优的Q值向量
        return self._filter_pareto_q_vectors(all_q_vectors)
    
    def _state_action_key(self, state: QuantitativeState, action: AgentAction) -> str:
        """生成状态-动作键"""
        return f"{state.to_key()}_{action.value}"
    
    def _filter_pareto_q_vectors(self, q_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """筛选帕累托最优Q值向量"""
        if not q_vectors:
            return []
        
        pareto_optimal = []
        for i, q1 in enumerate(q_vectors):
            is_dominated = False
            for j, q2 in enumerate(q_vectors):
                if i != j and self._q_dominates(q2, q1):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(q1)
        
        return pareto_optimal
    
    def _q_dominates(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        """判断q1是否支配q2"""
        return np.all(q1 >= q2) and np.any(q1 > q2)

# ===== 7. 权重学习器 =====
class WeightLearner:
    """动态权重学习器"""
    
    def __init__(self):
        self.weights = np.array([0.4, 0.25, 0.25, 0.1])  # [准确性, 效率, 完整性, 成本]
        self.adaptation_rate = 0.05
        self.performance_history = deque(maxsize=100)
    
    def update_weights(self, performance_feedback: Dict[str, float]):
        """根据性能反馈更新权重"""
        # 简单的权重适应规则
        if performance_feedback.get('accuracy_satisfaction', 0.5) < 0.6:
            self.weights[0] += self.adaptation_rate  # 增加准确性权重
            
        if performance_feedback.get('speed_satisfaction', 0.5) < 0.6:
            self.weights[1] += self.adaptation_rate  # 增加效率权重
            
        # 归一化权重
        self.weights = self.weights / np.sum(self.weights)
        
        # 记录性能历史
        self.performance_history.append(performance_feedback)
    
    def get_current_weights(self) -> np.ndarray:
        """获取当前权重"""
        return self.weights.copy()

# ===== 8. LLM智能决策器 =====
class LLMIntelligentDecisionMaker:
    """LLM智能决策器"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.decision_history = []
        
    def make_intelligent_decision(self, 
                                pareto_solutions: List[ParetoSolution],
                                rl_recommendation: AgentAction,
                                current_state: AgentState) -> ParetoSolution:
        """结合DP、RL、LLM的智能决策"""
        
        # 如果只有一个帕累托解，直接返回
        if len(pareto_solutions) <= 1:
            return pareto_solutions[0] if pareto_solutions else None
        
        # 构建决策prompt
        decision_prompt = self._build_decision_prompt(
            pareto_solutions, rl_recommendation, current_state
        )
        
        # LLM推理
        llm_response = self.llm_client.chat(
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": decision_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # 解析LLM决策
        try:
            decision_result = json.loads(llm_response.content)
            selected_solution = self._parse_llm_selection(decision_result, pareto_solutions)
        except Exception as e:
            print(f"LLM决策解析错误: {e}")
            # 降级到RL推荐
            selected_solution = self._find_solution_by_action(pareto_solutions, rl_recommendation)
        
        # 记录决策历史
        self._record_decision(pareto_solutions, selected_solution, decision_result)
        
        return selected_solution
    
    def _build_decision_prompt(self, 
                              pareto_solutions: List[ParetoSolution],
                              rl_recommendation: AgentAction,
                              current_state: AgentState) -> str:
        """构建决策prompt"""
        
        # 格式化帕累托解
        solutions_text = ""
        for i, solution in enumerate(pareto_solutions):
            solutions_text += f"""
方案 {i+1}:
- 动作: {solution.action.value}
- 准确性: {solution.objectives[0]:.3f}
- 效率: {solution.objectives[1]:.3f}
- 完整性: {solution.objectives[2]:.3f}
- 成本: {solution.objectives[3]:.3f}
- 期望奖励: {solution.expected_reward:.3f}
- 后续策略: {[a.value for a in solution.policy_path[:3]]}
"""
        
        prompt = f"""
你是一个智能Agent决策器。基于多目标动态规划，我们找到了以下帕累托最优解。
请选择最适合当前情况的方案。

=== 当前状态 ===
查询: {current_state.origin_query}
已执行动作: {[msg.action.value for msg in current_state.history[-3:] if msg.action]}
事实收集数: {len(current_state.collected_facts)}
时间消耗: {self._estimate_time_elapsed(current_state):.1f}%

=== 强化学习推荐 ===
基于历史经验，推荐动作: {rl_recommendation.value}

=== 帕累托最优方案 ===
{solutions_text}

=== 决策要求 ===
请综合考虑：
1. 当前查询的特点和复杂度
2. 已收集信息的充分性
3. 时间约束和效率要求
4. 用户对准确性vs速度的偏好
5. 强化学习的历史经验

请以JSON格式回复：
{{
    "selected_solution": 方案编号(1-{len(pareto_solutions)}),
    "reasoning": "选择理由(100字内)",
    "confidence": 置信度(0-1),
    "risk_assessment": "风险评估",
    "alternative": "备选方案编号"
}}
"""
        return prompt
    
    def _get_system_prompt(self) -> str:
        """获取系统prompt"""
        return """
你是一个专业的多目标决策专家，擅长在复杂的帕累托最优解中选择最适合的方案。
你的决策要综合考虑数学优化结果、历史经验和当前上下文。
保持决策的一致性和可解释性。
"""

# ===== 9. 混合智能控制器 =====
class HybridIntelligentController:
    """混合智能控制器 - 集成DP+RL+LLM"""
    
    def __init__(self):
        # 三大核心组件
        self.dp_solver = MultiObjectiveDynamicProgramming()
        self.rl_learner = MultiObjectiveQLearning()
        self.llm_decider = LLMIntelligentDecisionMaker()
        
        # 状态转换器
        self.state_converter = StateConverter()
        
        # 性能监控器
        self.performance_monitor = PerformanceMonitor()
        
    def decide_next_action(self, agent_state: AgentState) -> Message:
        """三层混合决策"""
        
        # Step 1: 转换为量化状态
        quant_state = self.state_converter.convert_to_quantitative(agent_state)
        
        # Step 2: 动态规划求解帕累托最优解集
        start_time = time.time()
        pareto_solutions = self.dp_solver.solve(quant_state)
        dp_time = time.time() - start_time
        
        print(f"DP求解耗时: {dp_time:.3f}秒, 找到 {len(pareto_solutions)} 个帕累托解")
        
        # Step 3: 强化学习推荐
        if pareto_solutions:
            available_actions = [sol.action for sol in pareto_solutions]
            rl_recommendation = self.rl_learner.select_action(quant_state, available_actions)
        else:
            # 如果DP没有找到解，RL自由选择
            all_actions = list(AgentAction)
            rl_recommendation = self.rl_learner.select_action(quant_state, all_actions)
        
        print(f"RL推荐动作: {rl_recommendation.value}")
        
        # Step 4: LLM智能最终决策
        if len(pareto_solutions) > 1:
            final_solution = self.llm_decider.make_intelligent_decision(
                pareto_solutions, rl_recommendation, agent_state
            )
        elif len(pareto_solutions) == 1:
            final_solution = pareto_solutions[0]
        else:
            # 降级方案：使用RL推荐
            final_solution = self._create_fallback_solution(rl_recommendation, quant_state)
        
        print(f"最终选择: {final_solution.action.value}")
        
        # Step 5: 创建决策消息
        decision_message = self._create_decision_message(final_solution, agent_state)
        
        # Step 6: 记录性能数据
        self.performance_monitor.record_decision(
            quant_state, final_solution, dp_time, len(pareto_solutions)
        )
        
        return decision_message
    
    def update_from_experience(self, 
                              previous_state: AgentState,
                              action_taken: AgentAction,
                              reward_received: np.ndarray,
                              new_state: AgentState):
        """从执行经验中学习"""
        
        # 转换状态
        prev_quant_state = self.state_converter.convert_to_quantitative(previous_state)
        new_quant_state = self.state_converter.convert_to_quantitative(new_state)
        
        # 更新RL组件
        experience = (prev_quant_state, action_taken, reward_received, new_quant_state)
        self.rl_learner.update_q_values(experience)
        
        # 更新权重学习器
        performance_feedback = self._calculate_performance_feedback(
            previous_state, new_state, reward_received
        )
        self.rl_learner.weight_learner.update_weights(performance_feedback)
        
        print(f"经验学习完成: 动作={action_taken.value}, 奖励={reward_received}")
    
    def _create_decision_message(self, solution: ParetoSolution, agent_state: AgentState) -> Message:
        """创建决策消息"""
        return Message(
            role=MessageRole.CONTROLLER,
            action=self._convert_to_message_action(solution.action),
            content={
                "query": agent_state.current_query,
                "strategy": "hybrid_optimization",
                "objectives_score": solution.objectives.tolist(),
                "policy_preview": [a.value for a in solution.policy_path[:3]]
            },
            metadata={
                "thought": f"基于多目标优化选择 {solution.action.value}",
                "confidence": solution.confidence,
                "expected_reward": solution.expected_reward,
                "optimization_method": "DP+RL+LLM",
                "pareto_optimal": True
            }
        )

# ===== 10. 状态转换器 =====
class StateConverter:
    """Agent状态与量化状态转换器"""
    
    def convert_to_quantitative(self, agent_state: AgentState) -> QuantitativeState:
        """将AgentState转换为QuantitativeState"""
        
        # 计算查询复杂度
        query_complexity = self._analyze_query_complexity(agent_state.origin_query)
        
        # 计算事实完整度
        facts_completeness = min(1.0, len(agent_state.collected_facts) / 5.0)
        
        # 计算搜索穷尽度
        search_actions = [msg for msg in agent_state.history 
                         if msg.action in [MessageAction.SEARCH_KG, MessageAction.SEARCH_VECTOR]]
        search_exhaustion = min(1.0, len(search_actions) / 3.0)
        
        # 计算时间消耗
        if agent_state.history:
            start_time = agent_state.history[0].timestamp
            current_time = time.time()
            elapsed = (current_time - start_time) / 300.0  # 假设总时间限制5分钟
            time_elapsed = min(1.0, elapsed)
        else:
            time_elapsed = 0.0
        
        # 计算置信度
        confidence_level = self._calculate_confidence_level(agent_state)
        
        return QuantitativeState(
            stage=len(agent_state.history),
            facts_completeness=facts_completeness,
            query_complexity=query_complexity,
            search_exhaustion=search_exhaustion,
            time_elapsed=time_elapsed,
            confidence_level=confidence_level
        )
    
    def _analyze_query_complexity(self, query: str) -> float:
        """分析查询复杂度"""
        if not query:
            return 0.0
        
        complexity_indicators = [
            ("compare", 0.3), ("analyze", 0.3), ("explain", 0.2),
            ("and", 0.1), ("or", 0.1), ("but", 0.1),
            ("what", 0.05), ("how", 0.15), ("why", 0.2)
        ]
        
        complexity = 0.3  # 基础复杂度
        query_lower = query.lower()
        
        for indicator, weight in complexity_indicators:
            if indicator in query_lower:
                complexity += weight
        
        # 基于长度的复杂度
        length_complexity = min(0.3, len(query.split()) / 20.0)
        complexity += length_complexity
        
        return min(1.0, complexity)
    
    def _calculate_confidence_level(self, agent_state: AgentState) -> float:
        """计算当前置信度"""
        if not agent_state.collected_facts:
            return 0.2
        
        # 基于事实数量和质量的置信度
        base_confidence = min(0.8, len(agent_state.collected_facts) / 5.0)
        
        # 基于最近工具执行结果的置信度调整
        recent_tool_msgs = [msg for msg in agent_state.history[-3:] 
                           if msg.role == MessageRole.TOOL]
        
        if recent_tool_msgs:
            avg_quality = np.mean([
                msg.metadata.get("quality_score", 0.5) 
                for msg in recent_tool_msgs
            ])
            base_confidence += 0.2 * avg_quality
        
        return min(1.0, base_confidence)

# ===== 11. 主系统集成 =====
class HybridOptimizedAgentSystem:
    """混合优化Agent系统"""
    
    def __init__(self):
        # 核心控制器
        self.controller = HybridIntelligentController()
        
        # 工具集
        self.tools = self._initialize_tools()
        
        # 监控组件
        self.performance_monitor = PerformanceMonitor()
        self.episode_tracker = EpisodeTracker()
        
    def process_query(self, user_query: str) -> str:
        """主处理流程"""
        print(f"\n=== 开始处理查询 ===")
        print(f"查询: {user_query}")
        
        # 初始化状态
        agent_state = AgentState()
        user_message = Message.create_user_query(user_query)
        agent_state.add_message(user_message)
        
        episode_states = [agent_state.copy()]
        max_iterations = 8
        
        for iteration in range(max_iterations):
            print(f"\n--- 迭代 {iteration + 1} ---")
            
            # 混合智能决策
            decision_message = self.controller.decide_next_action(agent_state)
            agent_state.add_message(decision_message)
            
            # 检查是否应该终止
            if self._should_terminate(agent_state, decision_message):
                print("达到终止条件")
                break
            
            # 执行工具
            if decision_message.action and decision_message.action != MessageAction.GENERATE:
                tool_result = self._execute_tool(decision_message, agent_state)
                agent_state.add_message(tool_result)
                
                # 从执行结果中学习
                self._learn_from_execution(agent_state, decision_message, tool_result)
            
            episode_states.append(agent_state.copy())
        
        # 生成最终答案
        final_answer = self._generate_final_answer(agent_state)
        
        # 记录完整episode
        self.episode_tracker.record_episode(episode_states, final_answer)
        
        print(f"\n=== 处理完成 ===")
        print(f"答案: {final_answer}")
        
        return final_answer
    
    def _learn_from_execution(self, agent_state: AgentState, decision: Message, result: Message):
        """从执行结果中学习"""
        if len(agent_state.history) < 2:
            return
        
        # 计算奖励向量
        reward_vector = self._calculate_reward_vector(decision, result, agent_state)
        
        # 获取前一状态
        previous_state_idx = -3  # decision之前的状态
        if len(agent_state.history) >= abs(previous_state_idx):
            # 构造前一状态（简化）
            previous_agent_state = self._reconstruct_previous_state(agent_state, previous_state_idx)
            
            # 更新控制器
            self.controller.update_from_experience(
                previous_agent_state,
                self._convert_message_action_to_agent_action(decision.action),
                reward_vector,
                agent_state
            )
    
    def _calculate_reward_vector(self, decision: Message, result: Message, state: AgentState) -> np.ndarray:
        """计算多目标奖励向量"""
        
        # 准确性奖励：基于工具结果质量
        accuracy_reward = result.metadata.get("quality_score", 0.5)
        
        # 效率奖励：基于执行时间
        execution_time = result.metadata.get("execution_time", 0.1)
        efficiency_reward = max(0.0, 1.0 - execution_time / 10.0)  # 假设10秒为上限
        
        # 完整性奖励：基于新增事实数量
        facts_before = len(state.collected_facts)
        new_facts = result.content.get("facts", [])
        completeness_reward = min(1.0, len(new_facts) / 3.0)
        
        # 成本奖励：基于资源消耗（负值）
        cost_penalty = -execution_time / 10.0
        
        return np.array([accuracy_reward, efficiency_reward, completeness_reward, cost_penalty])

# ===== 12. 主执行函数 =====
def main():
    """主执行函数"""
    
    # 初始化系统
    agent_system = HybridOptimizedAgentSystem()
    
    # 测试查询集
    test_queries = [
        "What films did Christopher Nolan direct and what are their main themes?",
        "Compare the economic growth of China and USA in the last decade",
        "Explain quantum computing and its potential applications in cryptography",
        "What are the causes and effects of climate change on Arctic ice?"
    ]
    
    print("混合优化Agent系统启动")
    print("结合多目标动态规划 + 强化学习 + LLM智能决策")
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"测试查询 {i+1}/{len(test_queries)}")
        print(f"{'='*60}")
        
        try:
            answer = agent_system.process_query(query)
            
            # 显示性能报告
            performance_report = agent_system.performance_monitor.get_latest_report()
            print(f"\n性能报告:")
            print(f"- DP求解时间: {performance_report.get('dp_time', 0):.3f}秒")
            print(f"- 帕累托解数量: {performance_report.get('pareto_count', 0)}")
            print(f"- 总执行时间: {performance_report.get('total_time', 0):.3f}秒")
            print(f"- 迭代次数: {performance_report.get('iterations', 0)}")
            
        except Exception as e:
            print(f"处理查询时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("所有测试完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()