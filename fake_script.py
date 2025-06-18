from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from langgraph.graph import StateGraph, END
import uuid
import time
import functools
from openai import OpenAI  # 或其他LLM客户端

# ===== 1. 消息结构 =====
@dataclass
class Message:
    role: str
    action: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 添加session_id用于追踪
    timestamp: float = field(default_factory=time.time)  # 添加时间戳便于调试和分析

# ===== 2. Agent状态 =====
@dataclass
class AgentState:
    query: str
    history: List[Message] = field(default_factory=list)
    current_message: Optional[Message] = None
    collected_facts: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, msg: Message) -> None:
        self.current_message = msg
        self.history.append(msg)
        
    def get_facts_by_source(self, source: str) -> List[Dict]:
        """获取特定来源的facts"""
        return [f for f in self.collected_facts if f["source"] == source]
    
    def get_last_message_by_role(self, role: str) -> Optional[Message]:
        """获取最近的特定角色消息"""
        return next((m for m in reversed(self.history) if m.role == role), None)

# ===== 3. 工具实现 =====
def decompose_tool(query: str) -> Dict[str, Any]:
    """查询分解工具,使用LLM将复杂查询分解为子查询"""
    # 实际实现中替换为真实的LLM调用
    return {
        "subqueries": [
            {
                "id": "sub1",
                "query": "What movies did Nolan direct between 2010-2020?",
                "reasoning": "分解时间范围,便于精确检索"
            },
            {
                "id": "sub2", 
                "query": "What are Nolan's most recent films?",
                "reasoning": "补充最新信息"
            }
        ],
        "strategy": "按时间维度拆分,确保信息完整性",
        "original_query": query
    }

def search_kg_tool(query: str) -> Dict[str, Any]:
    """知识图谱搜索工具,内部包含完整的检索链路"""
    
    def _align(query: str) -> Dict[str, Any]:
        """内部aligner,将查询对齐到图谱模式"""
        # 实际实现中替换为真实的LLM调用
        aligned_query = f"aligned: {query}"
        return {
            "kg_query": aligned_query,
            "confidence": 0.9,
            "metadata": {
                "reasoning": "将自然语言转换为图谱查询模式",
                "original_query": query
            }
        }
    
    def _execute(kg_query: str) -> Dict[str, Any]:
        """内部executor,执行图谱查询"""
        facts = [
            {"fact": "fact1", "confidence": 0.9, "source": "KG_Path_1"},
            {"fact": "fact2", "confidence": 0.85, "source": "KG_Path_2"}
        ]
        return {
            "facts": facts,
            "confidence": sum(f["confidence"] for f in facts) / len(facts),
            "metadata": {
                "kg_query": kg_query,
                "paths_searched": ["KG_Path_1", "KG_Path_2"]
            }
        }
    
    def _discriminate(facts: List[Dict], query: str) -> Dict[str, Any]:
        """内部discriminator,评估检索结果质量"""
        # 实际实现中替换为真实的LLM调用
        scores = [
            {
                "fact": fact["fact"],
                "relevance": 0.8,
                "reliability": 0.85,
                "completeness": 0.75
            }
            for fact in facts
        ]
        
        avg_score = sum(
            (s["relevance"] + s["reliability"] + s["completeness"]) / 3 
            for s in scores
        ) / len(scores)
        
        return {
            "score": avg_score,
            "fact_scores": scores,
            "metadata": {
                "query": query,
                "analysis": "详细分析每个fact的相关性、可靠性和完整性"
            }
        }

    # 内部检索链路执行
    try:
        # 1. 查询对齐
        align_result = _align(query)
        kg_query = align_result["kg_query"]
        
        # 2. 执行查询
        execute_result = _execute(kg_query)
        facts = execute_result["facts"]
        
        # 3. 结果评估
        evaluate_result = _discriminate(facts, query)
        
        # 4. 整合结果
        return {
            "facts": facts,
            "confidence": execute_result["confidence"],
            "quality_score": evaluate_result["score"],
            "metadata": {
                "query": query,
                "kg_query": kg_query,
                "align_confidence": align_result["confidence"],
                "fact_analysis": evaluate_result["fact_scores"],
                "execution_metadata": execute_result["metadata"]
            }
        }
    
    except Exception as e:
        # 异常处理
        return {
            "facts": [],
            "confidence": 0.0,
            "error": str(e),
            "metadata": {
                "query": query,
                "error_type": type(e).__name__
            }
        }

def search_vector_tool(query: str) -> Dict[str, Any]:
    """向量数据库搜索工具,内部包含完整的检索链路"""
    
    def _embed(query: str) -> Dict[str, Any]:
        """内部embedding,将查询转换为向量"""
        # 实际实现中替换为真实的embedding模型调用
        return {
            "vector": [0.1, 0.2, 0.3],  # 示例向量
            "confidence": 0.95,
            "metadata": {
                "model": "text-embedding-ada-002",
                "original_query": query
            }
        }
    
    def _search(vector: List[float]) -> Dict[str, Any]:
        """内部vector search,执行向量检索"""
        facts = [
            {"fact": "vector fact 1", "similarity": 0.92, "source": "Doc_1"},
            {"fact": "vector fact 2", "similarity": 0.88, "source": "Doc_2"}
        ]
        return {
            "facts": facts,
            "confidence": sum(f["similarity"] for f in facts) / len(facts),
            "metadata": {
                "top_k": 2,
                "index": "main_docs"
            }
        }
    
    def _rerank(facts: List[Dict], query: str) -> Dict[str, Any]:
        """内部reranker,重排序检索结果"""
        # 实际实现中替换为真实的reranker模型调用
        scores = [
            {
                "fact": fact["fact"],
                "relevance": 0.9,
                "semantic_similarity": 0.85
            }
            for fact in facts
        ]
        
        avg_score = sum(
            (s["relevance"] + s["semantic_similarity"]) / 2 
            for s in scores
        ) / len(scores)
        
        return {
            "score": avg_score,
            "fact_scores": scores,
            "metadata": {
                "query": query,
                "rerank_model": "cross-encoder"
            }
        }

    try:
        # 1. 生成查询向量
        embed_result = _embed(query)
        query_vector = embed_result["vector"]
        
        # 2. 执行向量检索
        search_result = _search(query_vector)
        facts = search_result["facts"]
        
        # 3. 结果重排序
        rerank_result = _rerank(facts, query)
        
        # 4. 整合结果
        return {
            "facts": facts,
            "confidence": search_result["confidence"],
            "quality_score": rerank_result["score"],
            "metadata": {
                "query": query,
                "embedding_confidence": embed_result["confidence"],
                "fact_analysis": rerank_result["fact_scores"],
                "search_metadata": search_result["metadata"],
                "rerank_metadata": rerank_result["metadata"]
            }
        }
    
    except Exception as e:
        # 异常处理
        return {
            "facts": [],
            "confidence": 0.0,
            "error": str(e),
            "metadata": {
                "query": query,
                "error_type": type(e).__name__
            }
        }

def generate_answer_tool(context: Dict[str, Any]) -> Dict[str, Any]:
    """答案生成工具,使用LLM基于收集的信息生成答案"""
    # context包含:原始query、分解strategy、子查询及推理过程、检索到的facts
    return {
        "answer": "根据调研,Christopher Nolan在2010-2020期间执导了...",
        "confidence": 0.95,
        "reasoning_chain": [
            "1. 分析了不同时期的作品",
            "2. 整合了最新信息",
            "3. 形成完整答案"
        ]
    }

# ===== 4. 工具注册表 =====
TOOL_REGISTRY = {
    "decompose": {
        "name": "decompose",
        "description": "将复杂查询分解为多个简单查询,并提供分解思路",
        "fn": decompose_tool,
        "required_fields": ["query"],
        "optional_fields": []
    },
    "search_kg": {
        "name": "search_kg",
        "description": "在知识图谱中搜索相关事实",
        "fn": search_kg_tool,
        "required_fields": ["query"],
        "optional_fields": []
    },
    "search_vector": {
        "name": "search_vector",
        "description": "在向量数据库中搜索相似内容",
        "fn": search_vector_tool,  # 使用完整的工具实现
        "required_fields": ["query"],
        "optional_fields": []
    },
    "generate": {
        "name": "generate",
        "description": "基于收集的信息生成最终答案",
        "fn": generate_answer_tool,
        "required_fields": ["context"],
        "optional_fields": []
    }
}

def validate_tool_input(tool_name: str, input_data: Any) -> bool:
    """工具输入验证"""
    tool = TOOL_REGISTRY[tool_name]
    if isinstance(input_data, dict):
        return all(field in input_data for field in tool["required_fields"])
    return isinstance(input_data, str)  # 默认接受字符串输入

# ===== 5. Controller逻辑 =====
def get_llm_decision(prompt: str) -> Dict[str, Any]:
    """调用LLM获取决策"""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": """你是一个智能Agent的控制器。
你需要基于当前状态决定下一步操作。
必须以JSON格式回复，包含:
{
    "thought": "推理过程",
    "action": "要使用的工具名",
    "input": "工具输入",
    "reason": "选择原因"
}"""
        }, {
            "role": "user",
            "content": prompt
        }],
        temperature=0.7,
        response_format={ "type": "json_object" }
    )
    return response.choices[0].message.content

def controller_node(state: AgentState) -> AgentState:
    """动态决策控制器"""
    
    # 1. 构建工具描述
    tools_desc = "\n".join(
        f"- {name}: {info['description']}\n  所需参数: {info['required_fields']}"
        for name, info in TOOL_REGISTRY.items()
    )
    
    # 2. 构建更丰富的上下文信息
    facts_summary = "\n".join(
        f"- Source: {fact['source']}\n  Facts: {fact['facts']}\n  Confidence: {fact.get('confidence', 'N/A')}"
        for fact in state.collected_facts
    )
    
    history_summary = "\n".join(
        f"Step {i}: {msg.role} -> {msg.action}\n  Result: {msg.content}\n  Thought: {msg.metadata.get('thought', 'N/A')}"
        for i, msg in enumerate(state.history)
    )
    
    current_status = {
        "facts_count": len(state.collected_facts),
        "steps_taken": len(state.history),
        "last_action": state.current_message.action if state.current_message else None,
        "last_result_quality": state.current_message.content.get("quality_score", 0) if state.current_message else 0
    }
    
    # 3. 构建提示词
    prompt = f"""当前查询: {state.query}

可用工具:
{tools_desc}

当前状态:
- 已收集Facts数量: {current_status['facts_count']}
- 已执行步骤: {current_status['steps_taken']}
- 上次行动: {current_status['last_action']}
- 上次结果质量: {current_status['last_result_quality']}

已收集的Facts:
{facts_summary}

执行历史:
{history_summary}

请分析当前状态并决定下一步行动:
1. 是否需要分解查询?
2. 是否需要收集更多信息?
3. 已有信息是否足够生成答案?
4. 是否需要尝试其他检索策略?

基于以上分析,给出你的决策。"""

    try:
        # 3. 调用LLM获取决策
        decision = get_llm_decision(prompt)
        
        # 4. 验证决策合法性
        if decision["action"] not in TOOL_REGISTRY and decision["action"] != "generate":
            raise ValueError(f"Invalid action: {decision['action']}")
        
        if not validate_tool_input(decision["action"], decision["input"]):
            raise ValueError(f"Invalid input for tool: {decision['action']}")
        
        # 5. 执行决策逻辑
        if decision["action"] == "generate":
            # 构建生成器所需的完整上下文
            generation_context = {
                "original_query": state.query,
                "decomposition": next(
                    (msg.content for msg in state.history 
                     if msg.role == "decompose"), 
                    None
                ),
                "collected_facts": state.collected_facts,
            }
            result = TOOL_REGISTRY["generate"]["fn"](generation_context)
            
            msg = Message(
                role="generator",
                action="generate",
                content=result,
                metadata={"thought": decision["thought"]}
            )
            state.add_message(msg)
            return END
        else:
            tool = TOOL_REGISTRY[decision["action"]]
            result = tool["fn"](decision["input"])
            
            msg = Message(
                role=tool["name"],
                action=decision["action"],
                content=result,
                metadata={
                    "thought": decision["thought"],
                    "reason": decision["reason"]
                }
            )
            state.add_message(msg)
            
            # 如果获取了新facts,保存起来
            if "facts" in result:
                state.collected_facts.append({
                    "source": decision["action"],
                    "facts": result["facts"],
                    "metadata": result.get("metadata", {})
                })
    
        return state
        
    except Exception as e:
        state.metadata["last_error"] = {
            "error": str(e),
            "context": "controller_decision",
            "prompt": prompt
        }
        return state

# ===== 6. 工具节点包装器 =====
def tool_node(state: AgentState, tool_name: str) -> AgentState:
    """工具节点的统一包装器"""
    try:
        tool = TOOL_REGISTRY[tool_name]
        result = tool["fn"](state.current_message.content)
        
        msg = Message(
            role=tool["name"],
            action=state.current_message.action,
            content=result,
            metadata=state.current_message.metadata
        )
        state.add_message(msg)
        
        if "facts" in result:
            state.collected_facts.append({
                "source": tool_name,
                "facts": result["facts"],
                "metadata": result.get("metadata", {})
            })
        
        # 记录成功状态
        state.metadata["last_error"] = None
        return state
        
    except Exception as e:
        # 记录错误信息
        state.metadata["last_error"] = {
            "tool": tool_name,
            "error": str(e),
            "error_type": type(e).__name__
        }
        return state

# ===== 7. 图构建 =====
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    # 1. 添加节点
    graph.add_node("controller", controller_node)
    graph.add_node("error_handler", error_handler_node)
    
    # 2. 添加工具节点
    for tool_name in TOOL_REGISTRY:
        graph.add_node(tool_name, functools.partial(tool_node, tool_name=tool_name))
    
    # 3. 设置入口点
    graph.set_entry_point("controller")
    
    # 4. 添加正常流程边
    for tool_name in TOOL_REGISTRY:
        graph.add_edge("controller", tool_name)
        graph.add_edge(tool_name, "controller")
    
    # 5. 添加错误处理边
    def has_error(state: AgentState) -> bool:
        """检查是否有错误"""
        return state.metadata.get("last_error") is not None
    
    def no_error(state: AgentState) -> bool:
        """检查是否无错误"""
        return not has_error(state)
    
    # 从工具节点到错误处理节点的条件边
    for tool_name in TOOL_REGISTRY:
        graph.add_conditional_edge(
            tool_name,
            "error_handler",
            condition=has_error
        )
    
    # 错误处理后返回controller
    graph.add_edge("error_handler", "controller")
    
    # 6. 添加结束条件
    def should_end(state: AgentState) -> bool:
        return (
            state.current_message and 
            (state.current_message.role == "generator" or
             len(state.history) > 10)  # 添加最大步数限制
        )
    
    graph.add_edge("controller", END, should_end)
    
    return graph.compile()

# ===== 8. 运行示例 =====
if __name__ == "__main__":
    query = "What films did Christopher Nolan direct?"
    state = AgentState(query=query)
    app = build_graph()
    final_state = app.invoke(state)
    
    print("\n🎯 Query:", query)
    print("\n📝 执行历史:")
    for msg in final_state.history:
        print(f"- [{msg.role}] {msg.action}")
        print(f"  思考: {msg.metadata.get('thought', '')}")
        print(f"  结果: {msg.content}")

def handle_error(state: AgentState, error_msg: str) -> AgentState:
    """统一的错误处理"""
    error_context = {
        "original_query": state.query,
        "error": error_msg,
        "collected_facts": state.collected_facts
    }
    
    result = {
        "answer": "抱歉，处理您的查询时遇到了问题。基于已收集的信息，我可以告诉您...",
        "confidence": 0.5,
        "error": error_msg
    }
    
    msg = Message(
        role="generator",
        action="generate",
        content=result,
        metadata={"error": error_msg}
    )
    state.add_message(msg)
    return END

def error_handler_node(state: AgentState) -> AgentState:
    """错误处理节点"""
    error_info = state.metadata.get("last_error", {})
    
    error_context = {
        "original_query": state.query,
        "error": error_info.get("error", "Unknown error"),
        "error_tool": error_info.get("tool", "Unknown tool"),
        "collected_facts": state.collected_facts
    }
    
    result = {
        "answer": f"抱歉，在使用 {error_context['error_tool']} 工具时遇到了问题。" +
                 "基于已收集的信息，我可以告诉您...",
        "confidence": 0.5,
        "error": error_context["error"]
    }
    
    msg = Message(
        role="error_handler",
        action="handle_error",
        content=result,
        metadata=error_context
    )
    state.add_message(msg)
    
    # 清除错误状态
    state.metadata["last_error"] = None
    
    return state
