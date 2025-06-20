"""
Agent Search System - 完整版：工具特定多目标动态规划+强化学习+LLM智能
结合三种方法：DP提供理论最优解，RL学习适应性策略，LLM提供智能决策
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import random

def AgentAction(value: str):
    """模拟AgentAction枚举"""
    return value

# ===== 1. 目标类型定义 =====
class ObjectiveType(Enum):
    """目标类型枚举"""
    # 通用目标
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency" 
    COMPLETENESS = "completeness"
    COST = "cost"
    
    # 分解特定目标
    CLARITY = "clarity"              
    COVERAGE = "coverage"            
    ACTIONABILITY = "actionability"  
    
    # 搜索特定目标
    RELEVANCE = "relevance"          
    AUTHORITY = "authority"          
    NOVELTY = "novelty"             
    RECALL = "recall"               
    PRECISION = "precision"         
    
    # 生成特定目标
    COHERENCE = "coherence"         
    FLUENCY = "fluency"            
    FACTUALITY = "factuality"      
    COMPREHENSIVENESS = "comprehensiveness"

# ===== 2. 增强的核心数据结构 =====
@dataclass
class QuantitativeState:
    """量化状态表示"""
    stage: int                      
    facts_completeness: float       
    query_complexity: float         
    search_exhaustion: float        
    time_elapsed: float             
    confidence_level: float         
    
    def to_key(self) -> str:
        """转换为状态键（用于记忆化）"""
        return f"{self.stage}_{self.facts_completeness:.1f}_{self.query_complexity:.1f}_{self.time_elapsed:.1f}_{self.search_exhaustion:.1f}_{self.confidence_level:.1f}"
    
    def to_vector(self) -> np.ndarray:
        """转换为向量形式（用于RL）"""
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
    objectives: np.ndarray               
    objective_types: List[ObjectiveType] 
    policy_path: List[AgentAction]       
    expected_reward: float               
    confidence: float                    
    tool_specific_score: float = 0.0     

@dataclass
class Experience:
    """强化学习经验"""
    state: QuantitativeState
    action: AgentAction
    objectives: np.ndarray              # 多目标奖励向量
    objective_types: List[ObjectiveType]
    next_state: QuantitativeState
    is_terminal: bool
    actual_transition_prob: float = 1.0

# ===== 3. 强化学习组件 =====
class MultiObjectiveQTable:
    """多目标Q表 - 存储每个状态-动作对的帕累托Q值集合"""
    
    def __init__(self, learning_rate: float = 0.1, gamma: float = 0.9):
        # Q(s,a) -> List[ParetoQValue] (帕累托Q值集合)
        self.q_table: Dict[str, Dict[AgentAction, List['ParetoQValue']]] = defaultdict(lambda: defaultdict(list))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.update_count = defaultdict(int)
    
    def get_pareto_q_values(self, state: QuantitativeState, action: AgentAction) -> List['ParetoQValue']:
        """获取状态-动作对的帕累托Q值集合"""
        state_key = state.to_key()
        return self.q_table[state_key][action]
    
    def update_q_values(self, experience: Experience, future_q_values: List['ParetoQValue']):
        """更新多目标Q值"""
        state_key = experience.state.to_key()
        action = experience.action
        
        # 计算目标Q值：immediate_reward + gamma * future_q_values
        target_q_values = []
        
        if experience.is_terminal:
            # 终止状态：只有即时奖励
            target_q_values = [ParetoQValue(
                objectives=experience.objectives,
                objective_types=experience.objective_types,
                visit_count=1
            )]
        else:
            # 非终止状态：即时奖励 + 折扣的未来奖励
            for future_q in future_q_values:
                # 组合当前奖励和未来Q值
                combined_objectives, combined_types = self._combine_objective_vectors(
                    experience.objectives, experience.objective_types,
                    future_q.objectives, future_q.objective_types
                )
                
                target_q_values.append(ParetoQValue(
                    objectives=combined_objectives,
                    objective_types=combined_types,
                    visit_count=1
                ))
        
        # 更新Q值：Q_new = (1-α) * Q_old + α * Q_target
        current_q_values = self.q_table[state_key][action]
        updated_q_values = self._update_pareto_q_set(current_q_values, target_q_values)
        
        self.q_table[state_key][action] = updated_q_values
        self.update_count[f"{state_key}_{action}"] += 1
    
    def _combine_objective_vectors(self, 
                                 immediate_obj: np.ndarray, immediate_types: List[ObjectiveType],
                                 future_obj: np.ndarray, future_types: List[ObjectiveType]) -> Tuple[np.ndarray, List[ObjectiveType]]:
        """组合即时奖励和未来Q值的目标向量"""
        all_objective_types = list(set(immediate_types + future_types))
        combined_vector = np.zeros(len(all_objective_types))
        
        # 映射即时奖励
        for i, obj_type in enumerate(immediate_types):
            global_index = all_objective_types.index(obj_type)
            combined_vector[global_index] += immediate_obj[i]
        
        # 映射未来Q值
        for i, obj_type in enumerate(future_types):
            if obj_type in all_objective_types:
                global_index = all_objective_types.index(obj_type)
                combined_vector[global_index] += self.gamma * future_obj[i]
        
        return combined_vector, all_objective_types
    
    def _update_pareto_q_set(self, current_q_set: List['ParetoQValue'], target_q_set: List['ParetoQValue']) -> List['ParetoQValue']:
        """更新帕累托Q值集合"""
        # 简化版本：直接合并并重新筛选帕累托最优
        all_q_values = current_q_set + target_q_set
        
        # 筛选帕累托最优的Q值
        return self._filter_pareto_q_values(all_q_values)
    
    def _filter_pareto_q_values(self, q_values: List['ParetoQValue']) -> List['ParetoQValue']:
        """筛选帕累托最优的Q值"""
        if not q_values:
            return []
        
        pareto_optimal = []
        for i, q1 in enumerate(q_values):
            is_dominated = False
            for j, q2 in enumerate(q_values):
                if i != j and self._q_value_dominates(q2, q1):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_optimal.append(q1)
        
        return pareto_optimal
    
    def _q_value_dominates(self, q1: 'ParetoQValue', q2: 'ParetoQValue') -> bool:
        """判断Q值1是否支配Q值2"""
        # 转换为统一目标空间进行比较
        unified_q1 = self._convert_q_to_unified_space(q1)
        unified_q2 = self._convert_q_to_unified_space(q2)
        
        all_geq = np.all(unified_q1 >= unified_q2)
        any_greater = np.any(unified_q1 > unified_q2)
        return all_geq and any_greater
    
    def _convert_q_to_unified_space(self, q_value: 'ParetoQValue') -> np.ndarray:
        """将Q值转换为统一目标空间 [accuracy, efficiency, completeness, cost]"""
        mapping = {
            ObjectiveType.CLARITY: ObjectiveType.ACCURACY,
            ObjectiveType.COVERAGE: ObjectiveType.COMPLETENESS,
            ObjectiveType.ACTIONABILITY: ObjectiveType.EFFICIENCY,
            ObjectiveType.RELEVANCE: ObjectiveType.ACCURACY,
            ObjectiveType.AUTHORITY: ObjectiveType.ACCURACY,
            ObjectiveType.NOVELTY: ObjectiveType.COMPLETENESS,
            ObjectiveType.RECALL: ObjectiveType.COMPLETENESS,
            ObjectiveType.PRECISION: ObjectiveType.ACCURACY,
            ObjectiveType.FACTUALITY: ObjectiveType.ACCURACY,
            ObjectiveType.COHERENCE: ObjectiveType.ACCURACY,
            ObjectiveType.COMPREHENSIVENESS: ObjectiveType.COMPLETENESS,
            ObjectiveType.FLUENCY: ObjectiveType.EFFICIENCY
        }
        
        unified = np.zeros(4)  # [accuracy, efficiency, completeness, cost]
        unified_counts = np.zeros(4)
        
        for i, obj_type in enumerate(q_value.objective_types):
            if obj_type == ObjectiveType.COST:
                unified[3] += q_value.objectives[i]
                unified_counts[3] += 1
            else:
                target_type = mapping.get(obj_type, ObjectiveType.ACCURACY)
                target_index = {
                    ObjectiveType.ACCURACY: 0,
                    ObjectiveType.EFFICIENCY: 1,
                    ObjectiveType.COMPLETENESS: 2,
                    ObjectiveType.COST: 3
                }[target_type]
                
                unified[target_index] += q_value.objectives[i]
                unified_counts[target_index] += 1
        
        # 归一化
        for i in range(4):
            if unified_counts[i] > 0:
                unified[i] /= unified_counts[i]
        
        return unified

@dataclass
class ParetoQValue:
    """帕累托Q值"""
    objectives: np.ndarray
    objective_types: List[ObjectiveType]
    visit_count: int = 0
    confidence: float = 1.0

class ExperienceBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, experience: Experience):
        """添加经验"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """采样经验批次"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def size(self) -> int:
        return len(self.buffer)

class ExplorationStrategy:
    """探索策略"""
    
    def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.1, epsilon_decay: float = 0.995):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0
    
    def should_explore(self) -> bool:
        """是否应该探索"""
        return random.random() < self.epsilon
    
    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.step_count += 1
    
    def get_exploration_action(self, feasible_actions: List[AgentAction]) -> AgentAction:
        """获取探索动作"""
        return random.choice(feasible_actions)

# ===== 4. LLM智能决策组件 =====
class LLMDecisionMaker:
    """LLM智能决策器"""
    
    def __init__(self):
        self.decision_history = []
        self.context_analyzer = ContextAnalyzer()
    
    def make_intelligent_decision(self, 
                                state: QuantitativeState,
                                dp_solutions: List[ParetoSolution],
                                rl_q_values: Dict[AgentAction, List[ParetoQValue]],
                                user_query: str = "") -> Tuple[AgentAction, str]:
        """基于DP解和RL经验的智能决策"""
        
        # 分析当前上下文
        context = self.context_analyzer.analyze_context(state, user_query)
        
        # 生成决策提示
        decision_prompt = self._generate_decision_prompt(state, dp_solutions, rl_q_values, context)
        
        # 模拟LLM决策过程
        decision, reasoning = self._simulate_llm_reasoning(decision_prompt)
        
        # 记录决策历史
        self.decision_history.append({
            'state': state,
            'decision': decision,
            'reasoning': reasoning,
            'dp_solutions_count': len(dp_solutions),
            'rl_coverage': len(rl_q_values)
        })
        
        return decision, reasoning
    
    def _generate_decision_prompt(self, 
                                state: QuantitativeState,
                                dp_solutions: List[ParetoSolution],
                                rl_q_values: Dict[AgentAction, List[ParetoQValue]],
                                context: Dict) -> str:
        """生成LLM决策提示"""
        
        prompt = f"""
        智能搜索决策场景:
        
        当前状态:
        - 阶段: {state.stage}
        - 事实完整度: {state.facts_completeness:.2f}
        - 查询复杂度: {state.query_complexity:.2f}
        - 搜索穷尽度: {state.search_exhaustion:.2f}
        - 时间消耗: {state.time_elapsed:.2f}
        - 置信度: {state.confidence_level:.2f}
        
        上下文分析:
        - 查询类型: {context.get('query_type', 'unknown')}
        - 紧急程度: {context.get('urgency', 'medium')}
        - 用户偏好: {context.get('user_preference', 'balanced')}
        
        动态规划建议的最优策略:
        """
        
        for i, sol in enumerate(dp_solutions[:3]):
            prompt += f"\n策略{i+1}: {sol.action} -> {sol.policy_path[:2]} (评分: {sol.tool_specific_score:.2f})"
        
        prompt += f"\n\n强化学习的经验:  "
        for action, q_vals in rl_q_values.items():
            if q_vals:
                avg_score = np.mean([np.sum(q.objectives) for q in q_vals])
                prompt += f"\n{action}: 平均Q值 {avg_score:.2f} (经验数: {len(q_vals)})"
        
        prompt += "\n\n请综合考虑理论最优性和实际经验，选择最佳动作。"
        
        return prompt
    
    def _simulate_llm_reasoning(self, prompt: str) -> Tuple[AgentAction, str]:
        """模拟LLM推理过程"""
        # 这里模拟LLM的推理逻辑
        # 实际实现中会调用真实的LLM API
        
        # 简化的决策逻辑：基于提示内容分析
        if "查询复杂度: 0.8" in prompt or "查询复杂度: 0.9" in prompt:
            if "阶段: 0" in prompt:
                return AgentAction.DECOMPOSE, "复杂查询应先分解以提高后续搜索效率"
            elif "事实完整度: 0.1" in prompt or "事实完整度: 0.2" in prompt:
                return AgentAction.SEARCH_KG, "事实不足，优先使用权威的知识图谱搜索"
        
        if "时间消耗: 0.8" in prompt or "时间消耗: 0.9" in prompt:
            return AgentAction.GENERATE, "时间紧迫，应尽快生成答案"
        
        if "事实完整度: 0.7" in prompt or "事实完整度: 0.8" in prompt:
            return AgentAction.GENERATE, "事实充分，可以生成高质量答案"
        
        # 默认策略
        return AgentAction.SEARCH_VECTOR, "平衡策略：使用向量搜索获取更多信息"

class ContextAnalyzer:
    """上下文分析器"""
    
    def analyze_context(self, state: QuantitativeState, user_query: str) -> Dict:
        """分析决策上下文"""
        context = {}
        
        # 分析查询类型
        if any(word in user_query.lower() for word in ['what', 'who', 'when', 'where']):
            context['query_type'] = 'factual'
        elif any(word in user_query.lower() for word in ['how', 'why', 'explain']):
            context['query_type'] = 'analytical'
        else:
            context['query_type'] = 'general'
        
        # 分析紧急程度
        if state.time_elapsed > 0.7:
            context['urgency'] = 'high'
        elif state.time_elapsed > 0.4:
            context['urgency'] = 'medium'
        else:
            context['urgency'] = 'low'
        
        # 分析用户偏好（基于状态推断）
        if state.query_complexity > 0.7:
            context['user_preference'] = 'accuracy_focused'
        elif state.time_elapsed > 0.5:
            context['user_preference'] = 'efficiency_focused'
        else:
            context['user_preference'] = 'balanced'
        
        return context

# ===== 5. 工具特定目标管理器（保持不变）=====
class ToolSpecificObjectives:
    """工具特定目标管理器"""
    
    def __init__(self):
        # 每个工具的特定目标
        self.tool_objectives = {
            AgentAction.DECOMPOSE: [
                ObjectiveType.CLARITY,
                ObjectiveType.COVERAGE, 
                ObjectiveType.ACTIONABILITY,
                ObjectiveType.COST
            ],
            AgentAction.SEARCH_KG: [
                ObjectiveType.RELEVANCE,
                ObjectiveType.AUTHORITY,
                ObjectiveType.RECALL,
                ObjectiveType.COST
            ],
            AgentAction.SEARCH_VECTOR: [
                ObjectiveType.RELEVANCE,
                ObjectiveType.NOVELTY,
                ObjectiveType.PRECISION,
                ObjectiveType.COST
            ],
            AgentAction.GENERATE: [
                ObjectiveType.FACTUALITY,
                ObjectiveType.COHERENCE,
                ObjectiveType.COMPREHENSIVENESS,
                ObjectiveType.FLUENCY
            ],
            AgentAction.REFINE: [
                ObjectiveType.ACCURACY,
                ObjectiveType.PRECISION,
                ObjectiveType.COHERENCE,
                ObjectiveType.COST
            ]
        }
        
        # 目标计算器
        self.objective_calculators = self._initialize_calculators()
        
        # 动态权重管理器
        self.weight_manager = AdaptiveObjectiveWeighting()
    
    def get_objectives_for_tool(self, action: AgentAction) -> List[ObjectiveType]:
        """获取工具的特定目标"""
        return self.tool_objectives.get(action, [
            ObjectiveType.ACCURACY,
            ObjectiveType.EFFICIENCY,
            ObjectiveType.COMPLETENESS,
            ObjectiveType.COST
        ]) #如果没有特定目标，则返回通用目标
    
    def calculate_tool_objectives(self, 
                                state: QuantitativeState, 
                                action: AgentAction) -> Tuple[np.ndarray, List[ObjectiveType]]:
        """计算工具特定的目标向量"""
        objectives = self.get_objectives_for_tool(action)
        objective_values = []
        
        for obj_type in objectives:
            calculator = self.objective_calculators[obj_type]
            value = calculator(state, action)
            objective_values.append(value)
        
        return np.array(objective_values), objectives
    
    def _initialize_calculators(self) -> Dict:
        """初始化目标计算器"""
        return {
            # 分解目标计算器
            ObjectiveType.CLARITY: self._calculate_clarity,
            ObjectiveType.COVERAGE: self._calculate_coverage,
            ObjectiveType.ACTIONABILITY: self._calculate_actionability,
            
            # 搜索目标计算器
            ObjectiveType.RELEVANCE: self._calculate_relevance,
            ObjectiveType.AUTHORITY: self._calculate_authority,
            ObjectiveType.NOVELTY: self._calculate_novelty,
            ObjectiveType.RECALL: self._calculate_recall,
            ObjectiveType.PRECISION: self._calculate_precision,
            
            # 生成目标计算器
            ObjectiveType.FACTUALITY: self._calculate_factuality,
            ObjectiveType.COHERENCE: self._calculate_coherence,
            ObjectiveType.COMPREHENSIVENESS: self._calculate_comprehensiveness,
            ObjectiveType.FLUENCY: self._calculate_fluency,
            
            # 通用目标计算器
            ObjectiveType.ACCURACY: self._calculate_accuracy,
            ObjectiveType.EFFICIENCY: self._calculate_efficiency,
            ObjectiveType.COMPLETENESS: self._calculate_completeness,
            ObjectiveType.COST: self._calculate_cost
        }
    
    # ===== 分解目标计算器 =====
    def _calculate_clarity(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算分解清晰度"""
        if action != AgentAction.DECOMPOSE:
            return 0.0
        
        base_clarity = 0.6
        complexity_bonus = 0.3 * state.query_complexity  # 复杂查询分解收益更大
        confidence_bonus = 0.1 * state.confidence_level
        
        return min(1.0, base_clarity + complexity_bonus + confidence_bonus)
    
    def _calculate_coverage(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算问题覆盖度"""
        if action != AgentAction.DECOMPOSE:
            return 0.0
        
        coverage = 0.5 + 0.4 * state.query_complexity
        
        # 如果已经分解过，覆盖度递减
        if state.stage > 1:
            coverage *= 0.8
            
        return min(1.0, coverage)
    
    def _calculate_actionability(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算可操作性"""
        if action != AgentAction.DECOMPOSE:
            return 0.0
        
        actionability = 0.7
        
        # 时间压力下，简单可操作的分解更有价值
        if state.time_elapsed > 0.5:
            actionability += 0.2 * (1 - state.time_elapsed)
        
        return min(1.0, actionability)
    
    # ===== 搜索目标计算器 =====
    def _calculate_relevance(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算搜索相关性"""
        if action not in [AgentAction.SEARCH_KG, AgentAction.SEARCH_VECTOR]:
            return 0.0
        
        base_relevance = 0.6
        
        if action == AgentAction.SEARCH_KG:
            # KG搜索对结构化查询相关性更高
            structural_bonus = 0.2 * (1 - state.query_complexity)
            return min(1.0, base_relevance + structural_bonus)
        else:  # SEARCH_VECTOR
            # 向量搜索对复杂语义查询相关性更高
            semantic_bonus = 0.3 * state.query_complexity
            return min(1.0, base_relevance + semantic_bonus)
    
    def _calculate_authority(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算信息权威性"""
        if action != AgentAction.SEARCH_KG:
            return 0.0
        
        authority = 0.8
        exhaustion_penalty = 0.2 * state.search_exhaustion
        
        return max(0.1, authority - exhaustion_penalty)
    
    def _calculate_novelty(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算信息新颖性"""
        if action != AgentAction.SEARCH_VECTOR:
            return 0.0
        
        base_novelty = 0.5
        novelty_potential = 0.4 * (1 - state.search_exhaustion)
        completeness_factor = 0.2 * (1 - state.facts_completeness)
        
        return min(1.0, base_novelty + novelty_potential + completeness_factor)
    
    def _calculate_recall(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算召回率"""
        if action != AgentAction.SEARCH_KG:
            return 0.0
        
        # KG搜索召回率通常较高，但随搜索深度递减
        recall = 0.8 - 0.3 * state.search_exhaustion
        return max(0.2, recall)
    
    def _calculate_precision(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算精确率"""
        if action not in [AgentAction.SEARCH_VECTOR, AgentAction.REFINE]:
            return 0.0
        
        if action == AgentAction.SEARCH_VECTOR:
            # 向量搜索精确率中等，但语义匹配好
            precision = 0.6 + 0.2 * (1 - state.query_complexity)
        else:  # REFINE
            # 精化操作提高精确率
            precision = 0.7 + 0.2 * state.confidence_level
        
        return min(1.0, precision)
    
    # ===== 生成目标计算器 =====
    def _calculate_factuality(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算事实准确性"""
        if action != AgentAction.GENERATE:
            return 0.0
        
        factuality = 0.3 + 0.6 * state.facts_completeness
        confidence_bonus = 0.1 * state.confidence_level
        
        return min(1.0, factuality + confidence_bonus)
    
    def _calculate_coherence(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算生成连贯性"""
        if action not in [AgentAction.GENERATE, AgentAction.REFINE]:
            return 0.0
        
        coherence = 0.8
        complexity_penalty = 0.2 * state.query_complexity
        
        return max(0.3, coherence - complexity_penalty)
    
    def _calculate_comprehensiveness(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算全面性"""
        if action != AgentAction.GENERATE:
            return 0.0
        
        return 0.4 + 0.5 * state.facts_completeness
    
    def _calculate_fluency(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算流畅性"""
        if action != AgentAction.GENERATE:
            return 0.0
        
        # LLM生成通常流畅性很好
        return 0.9
    
    # ===== 通用目标计算器（保持兼容性） =====
    def _calculate_accuracy(self, state: QuantitativeState, action: AgentAction) -> float:
        """通用准确性计算"""
        if action == AgentAction.DECOMPOSE:
            return self._calculate_clarity(state, action)
        elif action in [AgentAction.SEARCH_KG, AgentAction.SEARCH_VECTOR]:
            return self._calculate_relevance(state, action)
        elif action == AgentAction.GENERATE:
            return self._calculate_factuality(state, action)
        else:
            return 0.5
    
    def _calculate_efficiency(self, state: QuantitativeState, action: AgentAction) -> float:
        """通用效率计算"""
        base_efficiency = {
            AgentAction.DECOMPOSE: 0.8,
            AgentAction.SEARCH_KG: 0.5,
            AgentAction.SEARCH_VECTOR: 0.7,
            AgentAction.GENERATE: 0.9,
            AgentAction.REFINE: 0.4
        }.get(action, 0.5)
        
        time_penalty = 0.3 * state.time_elapsed
        return max(0.1, base_efficiency - time_penalty)
    
    def _calculate_completeness(self, state: QuantitativeState, action: AgentAction) -> float:
        """通用完整性计算"""
        if action in [AgentAction.SEARCH_KG, AgentAction.SEARCH_VECTOR]:
            return 0.6 * (1 - state.search_exhaustion)
        elif action == AgentAction.DECOMPOSE:
            return self._calculate_coverage(state, action)
        else:
            return 0.1
    
    def _calculate_cost(self, state: QuantitativeState, action: AgentAction) -> float:
        """通用成本计算（负值表示成本）"""
        base_costs = {
            AgentAction.DECOMPOSE: -0.1,
            AgentAction.SEARCH_KG: -0.2,
            AgentAction.SEARCH_VECTOR: -0.15,
            AgentAction.GENERATE: -0.3,
            AgentAction.REFINE: -0.2
        }
        
        base_cost = base_costs.get(action, -0.1)
        
        # 复杂度影响成本
        complexity_factor = 1 + 0.5 * state.query_complexity
        
        return base_cost * complexity_factor

    # ===== 自适应权重管理器 =====
class AdaptiveObjectiveWeighting:
    """自适应目标权重管理器"""
    
    def __init__(self):
        # 工具特定的基础权重
        self.base_weights = {
            AgentAction.DECOMPOSE: {
                ObjectiveType.CLARITY: 0.4,
                ObjectiveType.COVERAGE: 0.3,
                ObjectiveType.ACTIONABILITY: 0.2,
                ObjectiveType.COST: 0.1
            },
            AgentAction.SEARCH_KG: {
                ObjectiveType.RELEVANCE: 0.35,
                ObjectiveType.AUTHORITY: 0.35,
                ObjectiveType.RECALL: 0.2,
                ObjectiveType.COST: 0.1
            },
            AgentAction.SEARCH_VECTOR: {
                ObjectiveType.RELEVANCE: 0.4,
                ObjectiveType.NOVELTY: 0.3,
                ObjectiveType.PRECISION: 0.2,
                ObjectiveType.COST: 0.1
            },
            AgentAction.GENERATE: {
                ObjectiveType.FACTUALITY: 0.4,
                ObjectiveType.COHERENCE: 0.25,
                ObjectiveType.COMPREHENSIVENESS: 0.25,
                ObjectiveType.FLUENCY: 0.1
            }
        }
        
        self.adaptation_history = defaultdict(list)
        self.user_preferences = {}
    
    def get_dynamic_weights(self, 
                          action: AgentAction, 
                          state: QuantitativeState,
                          user_preference: Optional[Dict[str, float]] = None) -> Dict[ObjectiveType, float]:
        """获取动态调整的权重"""
        
        base_weights = self.base_weights.get(action, {}).copy()
        
        # 基于状态的权重调整
        if action == AgentAction.DECOMPOSE:
            if state.query_complexity > 0.7:
                # 复杂查询更重视覆盖度
                base_weights[ObjectiveType.COVERAGE] = base_weights.get(ObjectiveType.COVERAGE, 0.3) + 0.1
                base_weights[ObjectiveType.CLARITY] = base_weights.get(ObjectiveType.CLARITY, 0.4) - 0.1
        
        elif action == AgentAction.SEARCH_KG:
            if state.time_elapsed > 0.6:
                # 时间紧迫时更重视成本
                base_weights[ObjectiveType.COST] = base_weights.get(ObjectiveType.COST, 0.1) + 0.1
                base_weights[ObjectiveType.RECALL] = base_weights.get(ObjectiveType.RECALL, 0.2) - 0.1
        
        elif action == AgentAction.SEARCH_VECTOR:
            if state.facts_completeness < 0.4:
                # 事实不足时更重视新颖性
                base_weights[ObjectiveType.NOVELTY] = base_weights.get(ObjectiveType.NOVELTY, 0.3) + 0.15
                base_weights[ObjectiveType.RELEVANCE] = base_weights.get(ObjectiveType.RELEVANCE, 0.4) - 0.15
        
        # 融入用户偏好
        if user_preference:
            base_weights = self._incorporate_user_preference(base_weights, user_preference)
        
        # 归一化权重
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            for key in base_weights:
                base_weights[key] /= total_weight
        
        return base_weights
    
    def _incorporate_user_preference(self, weights: Dict, preferences: Dict) -> Dict:
        """融入用户偏好"""
        # 简化的偏好融入策略
        for obj_name, pref_weight in preferences.items():
            obj_type = ObjectiveType(obj_name) if obj_name in [e.value for e in ObjectiveType] else None
            if obj_type and obj_type in weights:
                weights[obj_type] *= (1 + pref_weight)
        
        return weights

# ===== 6. 混合智能决策系统 =====
class HybridIntelligentAgent:
    """混合智能决策系统：DP + RL + LLM"""
    
    def __init__(self):
        # 三个核心组件
        self.dp_solver = EnhancedMultiObjectiveDynamicProgramming()
        self.rl_system = MultiObjectiveRLAgent()
        self.llm_decision_maker = LLMDecisionMaker()
        
        # 决策融合策略
        self.fusion_strategy = "llm_guided"  # "dp_priority", "rl_priority", "voting", "llm_guided"
        
        # 性能统计
        self.decision_stats = {
            'dp_decisions': 0,
            'rl_decisions': 0, 
            'llm_decisions': 0,
            'hybrid_decisions': 0
        }
    
    def make_decision(self, state: QuantitativeState, user_query: str = "") -> Tuple[AgentAction, Dict]:
        """混合智能决策"""
        
        # 1. DP求解：获取理论最优策略
        dp_solutions = self.dp_solver.solve(state)
        
        # 2. RL推荐：获取学习的经验策略
        rl_recommendations = self.rl_system.get_action_recommendations(state)
        
        # 3. LLM智能决策：综合分析
        chosen_action, reasoning = self.llm_decision_maker.make_intelligent_decision(
            state, dp_solutions, rl_recommendations, user_query
        )
        
        # 4. 决策结果汇总
        decision_info = {
            'chosen_action': chosen_action,
            'reasoning': reasoning,
            'dp_solutions_count': len(dp_solutions),
            'rl_coverage': len(rl_recommendations),
            'dp_top_action': dp_solutions[0].action if dp_solutions else None,
            'rl_top_action': self._get_rl_top_action(rl_recommendations),
            'decision_confidence': self._calculate_decision_confidence(dp_solutions, rl_recommendations, chosen_action)
        }
        
        # 5. 统计更新
        self.decision_stats['hybrid_decisions'] += 1
        
        return chosen_action, decision_info
    
    def _get_rl_top_action(self, rl_recommendations: Dict[AgentAction, List[ParetoQValue]]) -> Optional[AgentAction]:
        """获取RL推荐的最佳动作"""
        if not rl_recommendations:
            return None
        
        best_action = None
        best_score = float('-inf')
        
        for action, q_values in rl_recommendations.items():
            if q_values:
                avg_score = np.mean([np.sum(q.objectives) for q in q_values])
                if avg_score > best_score:
                    best_score = avg_score
                    best_action = action
        
        return best_action
    
    def _calculate_decision_confidence(self, 
                                     dp_solutions: List[ParetoSolution],
                                     rl_recommendations: Dict[AgentAction, List[ParetoQValue]],
                                     chosen_action: AgentAction) -> float:
        """计算决策置信度"""
        confidence = 0.5  # 基础置信度
        
        # DP支持度
        if dp_solutions and chosen_action == dp_solutions[0].action:
            confidence += 0.3
        
        # RL支持度
        rl_top_action = self._get_rl_top_action(rl_recommendations)
        if rl_top_action and chosen_action == rl_top_action:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def learn_from_experience(self, experience: Experience):
        """从经验中学习"""
        self.rl_system.learn_from_experience(experience)
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'decision_stats': self.decision_stats,
            'rl_experience_count': self.rl_system.experience_buffer.size(),
            'dp_memo_size': len(self.dp_solver.pareto_memo),
            'llm_decision_history': len(self.llm_decision_maker.decision_history)
        }

class MultiObjectiveRLAgent:
    """多目标强化学习智能体"""
    
    def __init__(self):
        self.q_table = MultiObjectiveQTable()
        self.experience_buffer = ExperienceBuffer()
        self.exploration_strategy = ExplorationStrategy()
        self.tool_objectives = ToolSpecificObjectives()
        self.feasibility_checker = ActionFeasibilityChecker()
        
        self.learning_stats = {
            'episodes': 0,
            'total_updates': 0,
            'exploration_rate': 1.0
        }
    
    def get_action_recommendations(self, state: QuantitativeState) -> Dict[AgentAction, List[ParetoQValue]]:
        """获取基于Q学习的动作推荐"""
        feasible_actions = self._get_feasible_actions(state)
        recommendations = {}
        
        for action in feasible_actions:
            q_values = self.q_table.get_pareto_q_values(state, action)
            recommendations[action] = q_values
        
        return recommendations
    
    def choose_action(self, state: QuantitativeState) -> AgentAction:
        """选择动作（用于训练）"""
        feasible_actions = self._get_feasible_actions(state)
        
        if self.exploration_strategy.should_explore():
            # 探索：随机选择
            return self.exploration_strategy.get_exploration_action(feasible_actions)
        else:
            # 利用：选择最佳Q值动作
            return self._choose_best_action(state, feasible_actions)
    
    def _choose_best_action(self, state: QuantitativeState, feasible_actions: List[AgentAction]) -> AgentAction:
        """选择最佳动作"""
        best_action = feasible_actions[0]
        best_score = float('-inf')
        
        for action in feasible_actions:
            q_values = self.q_table.get_pareto_q_values(state, action)
            if q_values:
                # 计算平均Q值评分
                avg_score = np.mean([np.sum(q.objectives) for q in q_values])
                if avg_score > best_score:
                    best_score = avg_score
                    best_action = action
        
        return best_action
    
    def learn_from_experience(self, experience: Experience):
        """从经验中学习"""
        # 添加到经验缓冲区
        self.experience_buffer.add(experience)
        
        # 获取下一状态的Q值（用于更新）
        if not experience.is_terminal:
            next_state_q_values = []
            feasible_next_actions = self._get_feasible_actions(experience.next_state)
            for action in feasible_next_actions:
                q_vals = self.q_table.get_pareto_q_values(experience.next_state, action)
                next_state_q_values.extend(q_vals)
        else:
            next_state_q_values = []
        
        # 更新Q值
        self.q_table.update_q_values(experience, next_state_q_values)
        
        # 更新探索策略
        self.exploration_strategy.update_epsilon()
        
        # 统计更新
        self.learning_stats['total_updates'] += 1
        self.learning_stats['exploration_rate'] = self.exploration_strategy.epsilon
    
    def _get_feasible_actions(self, state: QuantitativeState) -> List[AgentAction]:
        """获取可行动作"""
        all_actions = [AgentAction.DECOMPOSE, AgentAction.SEARCH_KG, AgentAction.SEARCH_VECTOR, AgentAction.GENERATE, AgentAction.REFINE]
        return [action for action in all_actions if self.feasibility_checker.is_feasible(state, action)]

# ===== 7. 增强的动态规划求解器（保持不变但简化） =====
class EnhancedMultiObjectiveDynamicProgramming:
    """增强的多目标动态规划求解器"""
    
    def __init__(self):
        self.pareto_memo = {}
        self.state_transitions = StateTransitionModel()
        self.tool_objectives = ToolSpecificObjectives()
        self.feasibility_checker = ActionFeasibilityChecker()
        self.max_stages = 10
        self.gamma = 0.9
        self.solver_stats = {'memo_hits': 0, 'memo_misses': 0, 'pareto_solutions_generated': 0, 'objective_dimensions_used': defaultdict(int)}
    
    def solve(self, initial_state: QuantitativeState) -> List[ParetoSolution]:
        """主求解函数"""
        return self._solve_recursive(initial_state, self.max_stages)
    
    def _solve_recursive(self, state: QuantitativeState, stages_remaining: int) -> List[ParetoSolution]:
        """递归求解（简化版本）"""
        state_key = state.to_key()
        
        if state_key in self.pareto_memo:
            self.solver_stats['memo_hits'] += 1
            return self.pareto_memo[state_key]
        
        self.solver_stats['memo_misses'] += 1
        
        if stages_remaining == 0 or self._is_terminal_state(state):
            terminal_solutions = self._get_terminal_solutions(state)
            self.pareto_memo[state_key] = terminal_solutions
            return terminal_solutions
        
        feasible_actions = self._get_feasible_actions(state)
        if not feasible_actions:
            return self._get_terminal_solutions(state)
        
        all_candidate_solutions = []
        
        for action in feasible_actions:
            immediate_rewards, objective_types = self.tool_objectives.calculate_tool_objectives(state, action)
            next_state_distribution = self.state_transitions.get_next_states(state, action)
            
            for next_state, transition_prob in next_state_distribution:
                future_solutions = self._solve_recursive(next_state, stages_remaining - 1)
                
                for future_solution in future_solutions:
                    combined_objectives = immediate_rewards + self.gamma * transition_prob * future_solution.objectives[:len(immediate_rewards)]
                    
                    candidate_solution = ParetoSolution(
                        action=action,
                        state=state,
                        objectives=combined_objectives,
                        objective_types=objective_types,
                        policy_path=[action] + future_solution.policy_path,
                        expected_reward=np.sum(combined_objectives),
                        confidence=transition_prob * future_solution.confidence,
                        tool_specific_score=np.mean(combined_objectives)
                    )
                    
                    all_candidate_solutions.append(candidate_solution)
        
        pareto_optimal_solutions = self._filter_pareto_solutions(all_candidate_solutions)
        self.pareto_memo[state_key] = pareto_optimal_solutions
        return pareto_optimal_solutions
    
    def _filter_pareto_solutions(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """简化的帕累托筛选"""
        if not solutions:
            return []
        
        pareto_optimal = []
        for i, sol1 in enumerate(solutions):
            is_dominated = False
            for j, sol2 in enumerate(solutions):
                if i != j and np.all(sol2.objectives >= sol1.objectives) and np.any(sol2.objectives > sol1.objectives):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_optimal.append(sol1)
        
        return pareto_optimal
    
    def _get_feasible_actions(self, state: QuantitativeState) -> List[AgentAction]:
        all_actions = [AgentAction.DECOMPOSE, AgentAction.SEARCH_KG, AgentAction.SEARCH_VECTOR, AgentAction.GENERATE, AgentAction.REFINE]
        return [action for action in all_actions if self.feasibility_checker.is_feasible(state, action)]
    
    def _is_terminal_state(self, state: QuantitativeState) -> bool:
        return ((state.facts_completeness >= 0.8 and state.confidence_level >= 0.7) or state.time_elapsed >= 0.95 or state.stage >= self.max_stages)
    
    def _get_terminal_solutions(self, state: QuantitativeState) -> List[ParetoSolution]:
        terminal_rewards, terminal_types = self.tool_objectives.calculate_tool_objectives(state, AgentAction.GENERATE)
        return [ParetoSolution(action=AgentAction.GENERATE, state=state, objectives=terminal_rewards, objective_types=terminal_types,
                              policy_path=[AgentAction.GENERATE], expected_reward=np.sum(terminal_rewards), confidence=state.confidence_level, tool_specific_score=np.mean(terminal_rewards))]

# ===== 8. 状态转移和可行性检查（保持不变） =====
class StateTransitionModel:
    def get_next_states(self, current_state: QuantitativeState, action: AgentAction) -> List[Tuple[QuantitativeState, float]]:
        if action == AgentAction.DECOMPOSE:
            next_state = QuantitativeState(stage=current_state.stage + 1, facts_completeness=current_state.facts_completeness,
                                         query_complexity=max(0.2, current_state.query_complexity - 0.3), search_exhaustion=current_state.search_exhaustion,
                                         time_elapsed=current_state.time_elapsed + 0.1, confidence_level=current_state.confidence_level + 0.1)
            return [(next_state, 1.0)]
        elif action == AgentAction.SEARCH_KG:
            base_facts_gain, base_time_cost = 0.3, 0.2
            high_gain_state = QuantitativeState(stage=current_state.stage + 1, facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain + 0.2),
                                              query_complexity=current_state.query_complexity, search_exhaustion=min(1.0, current_state.search_exhaustion + 0.3),
                                              time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost), confidence_level=min(1.0, current_state.confidence_level + 0.2))
            low_gain_state = QuantitativeState(stage=current_state.stage + 1, facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain - 0.1),
                                             query_complexity=current_state.query_complexity, search_exhaustion=min(1.0, current_state.search_exhaustion + 0.2),
                                             time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost + 0.1), confidence_level=min(1.0, current_state.confidence_level + 0.05))
            return [(high_gain_state, 0.6), (low_gain_state, 0.4)]
        elif action == AgentAction.SEARCH_VECTOR:
            base_facts_gain, base_time_cost = 0.25, 0.15
            good_result_state = QuantitativeState(stage=current_state.stage + 1, facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain + 0.1),
                                                query_complexity=current_state.query_complexity, search_exhaustion=min(1.0, current_state.search_exhaustion + 0.25),
                                                time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost), confidence_level=min(1.0, current_state.confidence_level + 0.15))
            poor_result_state = QuantitativeState(stage=current_state.stage + 1, facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain - 0.05),
                                                query_complexity=current_state.query_complexity, search_exhaustion=min(1.0, current_state.search_exhaustion + 0.2),
                                                time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost + 0.05), confidence_level=min(1.0, current_state.confidence_level + 0.08))
            return [(good_result_state, 0.7), (poor_result_state, 0.3)]
        elif action == AgentAction.GENERATE:
            terminal_state = QuantitativeState(stage=current_state.stage + 1, facts_completeness=current_state.facts_completeness,
                                             query_complexity=current_state.query_complexity, search_exhaustion=current_state.search_exhaustion,
                                             time_elapsed=min(1.0, current_state.time_elapsed + 0.1), confidence_level=current_state.confidence_level)
            return [(terminal_state, 1.0)]
        return [(current_state, 1.0)]

class ActionFeasibilityChecker:
    def __init__(self):
        self.constraints = {'max_time': 0.95, 'min_confidence_for_generation': 0.6, 'max_search_exhaustion': 0.9, 'max_stages': 10}
    
    def is_feasible(self, state: QuantitativeState, action: AgentAction) -> bool:
        if (state.time_elapsed >= self.constraints['max_time'] or state.stage >= self.constraints['max_stages']):
            return action == AgentAction.GENERATE
        if action == AgentAction.DECOMPOSE:
            return state.query_complexity > 0.5 and state.stage <= 1
        elif action == AgentAction.SEARCH_KG:
            return state.search_exhaustion < self.constraints['max_search_exhaustion']
        elif action == AgentAction.SEARCH_VECTOR:
            return state.search_exhaustion < self.constraints['max_search_exhaustion']
        elif action == AgentAction.GENERATE:
            return (state.confidence_level >= self.constraints['min_confidence_for_generation'] or state.time_elapsed >= 0.8)
        elif action == AgentAction.REFINE:
            return state.facts_completeness > 0.3
        return True

# ===== 9. 主测试函数 =====
def main():
    """测试完整的混合智能系统"""
    
    print("完整混合智能系统测试：DP + RL + LLM")
    print("="*60)
    
    # 初始化混合智能系统
    agent = HybridIntelligentAgent()
    
    # 测试场景
    test_scenarios = [
        {
            'state': QuantitativeState(stage=0, facts_completeness=0.1, query_complexity=0.8, search_exhaustion=0.0, time_elapsed=0.0, confidence_level=0.3),
            'query': "What are the main causes of climate change?",
            'description': "复杂查询，初始阶段"
        },
        {
            'state': QuantitativeState(stage=2, facts_completeness=0.6, query_complexity=0.4, search_exhaustion=0.4, time_elapsed=0.5, confidence_level=0.7),
            'query': "How does artificial intelligence work?",
            'description': "中等复杂度，中期阶段"
        },
        {
            'state': QuantitativeState(stage=4, facts_completeness=0.8, query_complexity=0.3, search_exhaustion=0.7, time_elapsed=0.8, confidence_level=0.8),
            'query': "Who is the current president?",
            'description': "简单查询，后期阶段"
        }
    ]
    
    # 模拟学习过程
    print("模拟学习过程...")
    for i in range(10):
        # 生成随机经验用于RL学习
        random_state = QuantitativeState(stage=random.randint(0, 5), facts_completeness=random.random(), query_complexity=random.random(),
                                       search_exhaustion=random.random(), time_elapsed=random.random(), confidence_level=random.random())
        random_action = random.choice([AgentAction.DECOMPOSE, AgentAction.SEARCH_KG, AgentAction.SEARCH_VECTOR, AgentAction.GENERATE])
        objectives, obj_types = agent.rl_system.tool_objectives.calculate_tool_objectives(random_state, random_action)
        
        next_state_candidates = agent.rl_system.q_table.q_table['default'][random_action] = []
        
        experience = Experience(state=random_state, action=random_action, objectives=objectives, objective_types=obj_types,
                              next_state=random_state, is_terminal=random.random() > 0.7)
        agent.learn_from_experience(experience)
    
    print(f"学习完成，经验库大小: {agent.rl_system.experience_buffer.size()}")
    
    # 测试决策
    for i, scenario in enumerate(test_scenarios):
        print(f"\n--- 测试场景 {i+1}: {scenario['description']} ---")
        print(f"查询: {scenario['query']}")
        print(f"状态: 阶段{scenario['state'].stage}, 事实完整度{scenario['state'].facts_completeness:.1f}, 复杂度{scenario['state'].query_complexity:.1f}")
        
        # 混合决策
        start_time = time.time()
        chosen_action, decision_info = agent.make_decision(scenario['state'], scenario['query'])
        decision_time = time.time() - start_time
        
        print(f"\n决策结果:")
        print(f"- 选择动作: {chosen_action}")
        print(f"- 决策理由: {decision_info['reasoning']}")
        print(f"- 决策时间: {decision_time:.3f}秒")
        print(f"- 决策置信度: {decision_info['decision_confidence']:.2f}")
        print(f"- DP建议: {decision_info['dp_top_action']} (共{decision_info['dp_solutions_count']}个解)")
        print(f"- RL建议: {decision_info['rl_top_action']} (覆盖{decision_info['rl_coverage']}个动作)")
    
    # 系统状态报告
    print(f"\n--- 系统状态报告 ---")
    system_status = agent.get_system_status()
    print(f"决策统计: {system_status['decision_stats']}")
    print(f"RL经验数量: {system_status['rl_experience_count']}")
    print(f"DP缓存大小: {system_status['dp_memo_size']}")
    print(f"LLM决策历史: {system_status['llm_decision_history']}")
    
    print(f"\n{'='*60}")
    print("混合智能系统测试完成")

if __name__ == "__main__":
    main()