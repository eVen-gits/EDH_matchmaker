-- tournament definition

--CREATE TABLE tournament (
--    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT
--);

-- actions definition

--CREATE TABLE "actions" (
--    id INTEGER NOT NULL,
--    "before" INTEGER,
--    "after" INTEGER,
--    command TEXT,
--    "timestamp" TEXT DEFAULT (DATETIME('now')),
--    CONSTRAINT tournament_action_pk PRIMARY KEY (id),
--    FOREIGN KEY ("before") REFERENCES tournament(id),
--    FOREIGN KEY ("after") REFERENCES tournament(id)
--);

CREATE TABLE IF NOT EXISTS tournament_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    func_name TEXT NOT NULL,
    before_state TEXT,
    after_state TEXT,
    return_value TEXT,
    args TEXT,
    kwargs TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);