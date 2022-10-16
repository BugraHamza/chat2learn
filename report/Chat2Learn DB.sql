CREATE TYPE "SenderType" AS ENUM (
  'model',
  'user'
);

CREATE TABLE "user" (
  "id" int,
  "name" varchar,
  "lastname" varchar,
  "email" varchar,
  "password" varchar,
  "created_at" datetime
);

CREATE TABLE "model" (
  "id" int,
  "name" varchar
);

CREATE TABLE "chat_session" (
  "id" int,
  "user_id" int,
  "model_id" int,
  "hidden_state" blob,
  "cell_state" blob,
  "created_at" datetime
);

CREATE TABLE "message" (
  "id" int,
  "chat_session_id" int,
  "text" varchar,
  "sender" SenderType,
  "created_at" datetime
);

CREATE TABLE "report" (
  "id" int,
  "message_id" int,
  "correct_text" varchar
);

CREATE TABLE "report_error" (
  "report_id" int,
  "error_id" int
);

CREATE TABLE "error" (
  "id" int,
  "name" varchar,
  "description" varchar
);

ALTER TABLE "user" ADD FOREIGN KEY ("id") REFERENCES "chat_session" ("user_id");

ALTER TABLE "model" ADD FOREIGN KEY ("id") REFERENCES "chat_session" ("model_id");

ALTER TABLE "message" ADD FOREIGN KEY ("chat_session_id") REFERENCES "chat_session" ("id");

ALTER TABLE "report" ADD FOREIGN KEY ("message_id") REFERENCES "message" ("id");

ALTER TABLE "report" ADD FOREIGN KEY ("id") REFERENCES "report_error" ("report_id");

ALTER TABLE "error" ADD FOREIGN KEY ("id") REFERENCES "report_error" ("error_id");
