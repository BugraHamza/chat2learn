CREATE TYPE "SenderType" AS ENUM (
  'model',
  'user'
);

CREATE TABLE "person" (
  "id" int PRIMARY KEY,
  "name" varchar,
  "lastname" varchar,
  "email" varchar,
  "password" varchar,
  "created_at" datetime
);

CREATE TABLE "model" (
  "id" int PRIMARY KEY,
  "name" varchar
);

CREATE TABLE "chat_session" (
  "id" int PRIMARY KEY,
  "person_id" int,
  "model_id" int,
  "hidden_state" blob,
  "cell_state" blob,
  "created_at" datetime
);

CREATE TABLE "message" (
  "id" int PRIMARY KEY,
  "chat_session_id" int,
  "text" varchar,
  "sender" SenderType,
  "created_at" datetime
);

CREATE TABLE "report" (
  "id" int PRIMARY KEY,
  "message_id" int,
  "tagged_correct_text" varchar,
  "correct_text" varchar
);

CREATE TABLE "report_error" (
  "id" int PRIMARY KEY,
  "report_id" int,
  "error_code" varchar
);

ALTER TABLE "chat_session" ADD FOREIGN KEY ("person_id") REFERENCES "person" ("id");

ALTER TABLE "model" ADD FOREIGN KEY ("id") REFERENCES "chat_session" ("model_id");

ALTER TABLE "message" ADD FOREIGN KEY ("chat_session_id") REFERENCES "chat_session" ("id");

ALTER TABLE "report" ADD FOREIGN KEY ("message_id") REFERENCES "message" ("id");

ALTER TABLE "report_error" ADD FOREIGN KEY ("report_id") REFERENCES "report" ("id");
