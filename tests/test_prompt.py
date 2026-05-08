"""System prompt template — slots fill correctly, old hardcoded biases gone."""
from __future__ import annotations

from taxonomy_agent.prompts import SYSTEM_PROMPT_TEMPLATE


def _render(**overrides) -> str:
    base = dict(
        instruction="X", n_items=10, threshold=0.10, probe_size=20,
        max_iters=10, min_iters=3, size_aside="", focus_bullet="",
    )
    base.update(overrides)
    return SYSTEM_PROMPT_TEMPLATE.format(**base)


def test_template_no_longer_hardcodes_pattern_bias():
    """The hardcoded 'PATTERNS not topic content' line is gone."""
    rendered = _render()
    assert "PATTERNS" not in rendered
    assert "not topic content" not in rendered


def test_default_rendering_omits_focus_bullet():
    rendered = _render()
    assert "Categories should describe" not in rendered


def test_focus_bullet_appears_when_set():
    rendered = _render(
        focus_bullet="- Categories should describe what each text is about.\n"
    )
    assert "- Categories should describe what each text is about." in rendered


def test_size_aside_appears_when_set():
    rendered = _render(size_aside=" (aim for 10–25 categories)")
    assert "aim for 10–25 categories" in rendered


def test_size_aside_omitted_when_template_slot_empty():
    """Template-level: an empty `size_aside` slot does not introduce 'aim for'."""
    rendered = _render()
    assert "aim for" not in rendered


def test_no_target_size_aside_renders():
    """When the run() helper produces the 'no target' aside (size_hint empty/None),
    the orchestrator is told there is no fixed count to aim for."""
    rendered = _render(size_aside=" (use whatever number of categories fits the corpus)")
    assert "use whatever number of categories fits the corpus" in rendered
    assert "aim for" not in rendered


def test_constraints_section_intact_after_focus_bullet_removal():
    """Even without a focus bullet, the rest of Constraints renders."""
    rendered = _render()
    assert "Names: short snake_case" in rendered
    assert "JSON-only replies" in rendered


def test_min_iters_appears_in_stop_rule_and_loop():
    """The floor on classification rounds is mentioned in both the high-level
    stop rule and the per-iteration loop guidance, so the orchestrator can't
    miss it."""
    rendered = _render(min_iters=5)
    # Mentioned at least twice — once in the "Your job" stop rule, once in the
    # iteration loop's convergence step.
    assert rendered.count("5 classification rounds") + rendered.count("5 classify rounds") >= 2


def test_strategies_focus_renders_naturally():
    rendered = _render(
        focus_bullet="- Categories should describe the reasoning strategy "
                     "each chain of thought uses.\n"
    )
    assert "the reasoning strategy each chain of thought uses" in rendered
