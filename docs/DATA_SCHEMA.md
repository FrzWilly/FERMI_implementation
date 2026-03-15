# DATA SCHEMA（Phase 1）

統一樣本欄位（`UnifiedSample`）：

- `id`: str
- `task`: str
- `split`: str
- `input_text`: str
- `profile`: list[dict]
- `gold`: str | int | None
- `user_id`: str（由 profile id 雜湊推導）

