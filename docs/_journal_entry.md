

## analyze best.pt(8621, depth=20): val 9.1295 (new best), bus растёт, MLP спит, active_depth НЕ пишется (2026-08-27)

- depth=20 (нет сообщения 24/24, mem 12.5GB). val_loss=9.1295 (ppl 9223.59) - new best (лучше 9.1686 и 10.53).
- Bridge МЕДЛЕННО РАСТЁТ (впервые с 8388, ранее был плоским): BUS(_last_bus) norm 3259->4828,
  per-expert 570->838; L0 w_intent 0.1418->0.1435; intent_probe W_norm 19.59->19.67;
  bus_head_proj 0.6515->0.6601. cross-layer cosine 0.007->0.054 (всё ещё низкий => слои
  дополняют, не дублируют). => intent-шина наращивает пропускную способность по мере обучения.
- MLP СПИТ: sigmoid(mod_scale_mlp)=0.661, W_std dev +0.0007 (без изменений с 7922). pred head
  МЁРТВ (requires_grad=False). L12/L23 w_intent плоские (0.0346/0.0164). slots 87->89/192.
  births экспертов в [10,11,12,13,15] (неизменно в середине).
- ВАЖНО: best.pt(8621) НЕ содержит ключа 'active_depth' (top-level keys: step, model, optimizer,
  scheduler, cfg, best_val_loss, param_names, reasoning_enabled_step). Значит save-часть фикса в
  ЗАПУЩЕННОМ тренинге не пишет глубину -> следующий резюм (если не запатчить) даст фоллбэк=8 и
  реклимб. Причина: либо ноутбук в Colab - старая копия без save-фикса, либо save-блок отличается.
- РЕКОМЕНДАЦИЯ: перед следующим резюмом вшить active_depth=20 в best.pt (и последний step_*.pt) на
  Drive; ИЛИ перезагрузить полностью зафиксенный notebooks/colab.ipynb (репо-версия пишет
  active_depth во все 3 saves) -> тогда saves автоматически сохраняют глубину и ручной патч не нужен.
