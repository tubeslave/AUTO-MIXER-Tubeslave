# Agent Responsibilities

> Этот шаблонный слой создаётся отдельно от текущих runtime-агентов
> проекта и не вмешивается в `backend/agents`.

## mixing_agent
Отвечает за сведение (gain, EQ, compression, routing)

## architect_agent
Проектирует систему агентов и их взаимодействие

## trainer_agent
Организует обучение моделей и пайплайны

## evaluator_agent
Проверяет качество (метрики + субъективная оценка)

## coordinator
Оркестрирует агентов и распределяет задачи
