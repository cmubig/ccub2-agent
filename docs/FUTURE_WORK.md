# CCUB2 Agent - Future Work & Architecture Decisions

This document records our architectural discussions, options considered, decisions made, and rationale. Updated chronologically as we make key decisions.

---

## 2025-10-31: Multi-Agent Architecture Discussion

### Context
After reading Anthropic's "Building Effective Agents" guide, we reconsidered whether CCUB2 should be a workflow system or multi-agent system.

### Current State Analysis
- **Architecture**: Single Agent + Workflow
- **Pattern**: Main Agent orchestrates pre-defined pipeline (T2I → VLM → I2I loop)
- **Classification**: 90% Workflow, 10% Agentic

### Options Considered

#### Option 1: Keep Current Workflow
- Pros: Already working, simple, fast to complete
- Cons: Limited flexibility, hard to scale, no autonomous planning

#### Option 2: Hybrid (Single Coordinator + Specialist Agents)
- Pros: Better reasoning, separation of concerns, easier to extend
- Cons: More complex, 2-3 weeks additional work

#### Option 3: Full Multi-Agent System
- Pros: Maximum flexibility, true autonomy, best long-term
- Cons: Major refactoring (2-3 months), premature at this stage

### Decision: Phased Approach ✅

**Phase 1 (NOW)**: Complete current workflow system
- Finish knowledge extraction
- Test pipeline end-to-end
- Measure baseline performance

**Phase 2 (Later)**: Gradually introduce specialist models
- Keep single orchestrator
- Add specialized reasoning if needed
- Maintain backward compatibility

**Phase 3 (Future)**: Evolve to multi-agent when needed
- Split into specialist agents
- Add inter-agent communication
- Implement learning mechanisms

### Rationale
1. **Pragmatic**: Current system ~75% complete, finish first
2. **Measurable**: Need baseline before adding complexity
3. **Reversible**: Can add agents incrementally
4. **Risk management**: Don't over-engineer before validation

### Reference
- Anthropic Guide: https://www.anthropic.com/engineering/building-effective-agents
- Key Insight: "Start simple, add complexity only when simpler solutions fall short"

---

## 2025-10-31: Reasoning Model Selection

### Problem
Should we use a separate reasoning model for cultural analysis, or keep everything in the VLM?

### Options Evaluated

#### 1. DeepSeek-R1 Series
- **Sizes**: 671B (too large), 14B/7B distilled versions
- **Vision**: ❌ NO (text-only)
- **Reasoning**: ✅ Excellent (RL-trained, thinking tokens)
- **Korean**: ⚠️ Limited
- **Use case**: Text reasoning, but needs vision wrapper

#### 2. Qwen3-VL-8B-Thinking
- **Size**: 8B
- **Vision**: ✅ YES
- **Reasoning**: ✅ Good (thinking tokens)
- **Korean**: ✅ Excellent
- **Status**: Exists on HuggingFace, released January 2025

#### 3. QwQ-32B (Qwen with Questions)
- **Size**: 32B (4-bit = 16GB)
- **Vision**: ❌ NO
- **Reasoning**: ✅ Excellent (o1-style)
- **Korean**: ✅ Good (Qwen family)
- **Use case**: Pure reasoning, needs separate vision stage

#### 4. gpt-oss (OpenAI)
- **Sizes**: 120B / 20B
- **Vision**: ❌ NO
- **Reasoning**: ✅ Excellent
- **Korean**: ⚠️ OK
- **Cons**: Too large, no vision

#### 5. Current: Qwen3-VL-8B-Instruct
- **Size**: 8B ✅
- **Vision**: ✅ YES
- **Reasoning**: ⚠️ Implicit (no thinking tokens)
- **Korean**: ✅ Excellent
- **Status**: Already working

### Comparison Matrix

| Model | Vision | Reasoning | Size | Korean | Deployment | Best For |
|-------|--------|-----------|------|--------|------------|----------|
| Qwen3-VL-8B-Instruct | ✅ | ⚠️ Implicit | 8B | ✅ | Easy | **Current** |
| Qwen3-VL-8B-Thinking | ✅ | ✅ Explicit | 8B | ✅ | Easy | **Best upgrade** |
| DeepSeek-R1-14B | ❌ | ✅ Excellent | 14B | ⚠️ | Medium | Text reasoning |
| QwQ-32B | ❌ | ✅ Excellent | 32B* | ✅ | Hard | Text reasoning |
| gpt-oss-20b | ❌ | ✅ Excellent | 20B | ⚠️ | Hard | Text reasoning |

*With 4-bit quantization

### Key Finding
**Qwen3-VL-8B-Thinking is the ideal model**: Vision + Reasoning + Korean + 8B size!

All other reasoning models (DeepSeek-R1, QwQ, gpt-oss) are **text-only** and would require a separate vision stage.

### Architecture Implications

#### Option A: VLM-Only (Unified Model)
```
Qwen3-VL-8B-Instruct or Qwen3-VL-8B-Thinking
    ↓
Vision Analysis + Cultural Reasoning
    ↓
Result
```
- **Pros**: Simple, fast, less memory, already working
- **Cons**: Reasoning quality limited (Instruct), better with Thinking version

#### Option B: VLM + LLM (Separated)
```
Qwen3-VL-8B (Vision)
    ↓
QwQ-32B or DeepSeek-R1 (Reasoning)
    ↓
Result
```
- **Pros**: Better reasoning, separation of concerns, future-proof
- **Cons**: Complex, more memory, latency, info loss in vision→text

### Decision: VLM-Only First, Consider Thinking Model ✅

**Immediate (Weeks 1-2)**:
- Keep Qwen3-VL-8B-Instruct
- Enhance with better prompting (explicit reasoning steps)
- Complete current pipeline
- Measure baseline performance

**Quick Win (Week 3)**:
- Test Qwen3-VL-8B-Thinking (drop-in replacement)
- Compare: Instruct vs Thinking on same images
- If better reasoning: Switch to Thinking version
- **Advantage**: Same 8B size, no architecture change!

**If Still Insufficient (Weeks 4+)**:
- Try Qwen3-VL + QwQ-32B separated pipeline
- Compare performance: VLM-only vs VLM+LLM
- Decide based on results

### Rationale

1. **Perfect model exists!**: Qwen3-VL-8B-Thinking has vision + reasoning + Korean + 8B
2. **Drop-in upgrade**: Just change model name, no code refactoring
3. **Measurable**: Test Instruct → Thinking first before separating models
4. **Pragmatic**: Don't separate until proven necessary
5. **Future-ready**: If we separate later, already thinking in specialist terms

### User's Key Insight
> "나중에 에이전트로 나눌 거니까 지금부터 분리하는 게 맞지 않나?"

**Response**: YES, but gradually:
1. First: Complete current system (get baseline)
2. Then: Try Thinking version (easy upgrade)
3. Then: Separate if reasoning quality still insufficient
4. Already separated = easy multi-agent later

### Quote from Discussion
> "VLM에 다 시키기 vs 역할 분리?"

**Answer**:
- **Now**: VLM does everything (simple, fast)
- **Soon**: Try VLM-Thinking (better reasoning, still unified)
- **Later**: Separate vision/reasoning if quality demands it
- **Future**: Full multi-agent when scaling requires it

---

## Next Steps (Priority Order)

### Immediate (This Week)
1. ✅ Scoring scale 1-5 → 1-10 (COMPLETED 2025-10-31)
2. ⏳ Extract cultural knowledge from images
3. ⏳ Test complete pipeline end-to-end
4. ⏳ Measure baseline: detection accuracy, iteration count, quality

### Short-term (Weeks 2-3)
1. Enhance VLM prompting with explicit reasoning steps
2. Optimize RAG knowledge injection
3. Test with diverse image set (20+ images)
4. Document baseline performance

### Quick Win Test (Week 3)
1. Test Qwen3-VL-8B-Thinking as drop-in replacement
2. Compare Instruct vs Thinking:
   - Detection accuracy
   - Reasoning depth (thinking tokens visible)
   - Explanation quality
   - Korean understanding
3. If better: Switch to Thinking version (no code changes!)

### Medium-term (Months 2-3)
1. **If Thinking sufficient**: Optimize current system
2. **If reasoning still lacking**: Evaluate VLM+LLM separation
   - Test QwQ-32B-4bit as reasoning specialist
   - Compare VLM-only vs VLM+LLM on metrics
   - Decide architecture based on results

### Long-term (Months 3-6)
1. Multi-agent evolution (if needed):
   - Vision Analysis Agent
   - Cultural Expert Agent
   - Editing Strategy Agent
   - Research Agent
   - QA Agent
2. Learning/memory system
3. Agent communication protocol

---

## Open Questions

### Technical
1. **Qwen3-VL-8B-Thinking performance?**
   - How much better than Instruct version?
   - Does thinking process slow down inference?
   - Are thinking tokens useful for debugging?

2. **Memory constraints for VLM+LLM?**
   - 8B VLM + 32B LLM (4-bit) = ~24GB
   - Can we unload VLM when using reasoning model?
   - Need to benchmark actual memory usage

3. **Information loss in vision→text conversion?**
   - How much visual detail lost when VLM describes to text?
   - Can structured JSON output minimize loss?
   - Test with side-by-side comparison

### Architectural
1. **When is multi-agent worth the complexity?**
   - What performance improvement justifies the overhead?
   - +10% accuracy? +20%? Explainability?
   - Need clear success criteria before committing

2. **How to handle model updates?**
   - New reasoning models released frequently
   - Should architecture be model-agnostic?
   - Version pinning vs latest models?

3. **Korean language support priority?**
   - All reasoning models (DeepSeek, gpt-oss) have limited Korean
   - Is Qwen family's Korean support critical?
   - Could we fine-tune reasoning models for Korean?

### Process
1. **How to measure "reasoning quality"?**
   - Accuracy is clear, but reasoning depth?
   - Human eval needed? Or proxy metrics?
   - Develop rubric for reasoning assessment

2. **Baseline thresholds?**
   - What accuracy triggers architecture change?
   - <70%? <80%? User satisfaction vs metric?
   - Define success criteria upfront

---

## Lessons Learned

### From Anthropic Guide
1. **Start simple**: Most successful implementations use simple patterns
2. **Avoid frameworks initially**: Direct LLM API calls more transparent
3. **Building blocks → Workflows → Agents**: Natural progression
4. **Measure before complexifying**: Don't add agents until proven needed

### From Our Discussion
1. **Perfect models DO exist**: Qwen3-VL-8B-Thinking has vision+reasoning+Korean+8B
2. **Trade-offs are real**: But sometimes you get lucky with model releases
3. **Incremental is better**: Complete → Measure → Improve → Repeat
4. **Future-proof thinking**: Design for separation even if unified now

### User Preferences Identified
1. **Pragmatic over perfect**: Finish working system first
2. **Forward-thinking**: Consider multi-agent future in current design
3. **Evidence-based**: Measure before adding complexity
4. **Open-source focus**: Local deployment, cost-effective, customizable

---

## References

### Articles & Papers
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs](https://arxiv.org/abs/2501.xxxxx)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)

### Models Evaluated
- [Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) ⭐ **Ideal choice**
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (current)
- [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- [QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview)
- [OpenAI gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

### Related Work
- IBM: "What Is a Reasoning Model?"
- LangGraph (LangChain agent framework)
- Amazon Bedrock AI Agent framework

---

## Timeline of Decisions

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| 2025-10-31 | Scoring: 1-5 → 1-10 | Finer granularity for evaluation | ✅ Completed |
| 2025-10-31 | Architecture: Workflow-first, multi-agent later | Complete current system before complexifying | ✅ Decided |
| 2025-10-31 | Models: Keep Qwen3-VL-8B-Instruct, test Thinking next | Measure baseline, then easy upgrade | ✅ Decided |
| TBD | Instruct vs Thinking comparison | Depends on baseline + Thinking test | ⏳ Week 3 |
| TBD | VLM-only vs VLM+LLM | Depends on Thinking model performance | ⏳ If needed |
| TBD | Multi-agent transition | Depends on scaling needs | ⏳ Future |

---

## Experimental Plan: Testing Qwen3-VL-8B-Thinking

### Hypothesis
Qwen3-VL-8B-Thinking will provide better cultural reasoning than Instruct version while maintaining same vision quality and deployment ease.

### Test Protocol (Week 3)

1. **Same infrastructure**: No code changes, just model swap
2. **Test set**: 20 images (10 with issues, 10 correct)
3. **Metrics**:
   - Detection accuracy (issue detection rate)
   - False positive rate
   - Reasoning depth (thinking token analysis)
   - Korean understanding quality
   - Inference time difference

### Expected Results

**If Thinking >> Instruct**:
- Switch to Thinking permanently
- Document reasoning improvements
- Benefit from visible thinking process

**If Thinking ≈ Instruct**:
- Keep Instruct (simpler)
- Or keep Thinking if thinking tokens useful for debugging

**If Thinking < Instruct**:
- Stay with Instruct
- Consider VLM+LLM separation

---

## Contact & Updates

This document will be updated as we make progress and new decisions. Each session should add:
1. Date
2. Topic discussed
3. Options considered
4. Decision made
5. Rationale
6. Status/next steps

**Last Updated**: 2025-10-31
**Next Review**: After baseline testing (Week 2)

---

*"Start simple, measure, then iterate. Don't add complexity until simpler solutions fall short."* — Anthropic
