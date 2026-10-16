# 云雀智能客服系统 · API 接口规范

## 通用约定

所有接口的基础地址为 `https://api.skylark.example.com/v3`。请求与响应均使用 JSON 格式，编码为 UTF-8。

## 鉴权方式

接口采用 Bearer Token 鉴权。调用方需在请求头中携带 `Authorization: Bearer <access_token>`。

access_token 通过 `POST /auth/token` 获取，请求体需要提供 client_id 与 client_secret。token 的有效期为 7200 秒，过期后需要用 refresh_token 换取新 token，refresh_token 有效期为 30 天。

请勿在客户端（浏览器、移动 App）中硬编码 client_secret，一旦泄露视为安全事故。正确的做法是由自己的服务端代理调用。

## 限流规则

接口按 client_id 维度限流，默认每分钟 600 次请求，突发允许到 900 次。

超出限流配额时返回 HTTP 429，响应体会带上 `Retry-After` 头，单位是秒。调用方应当实现指数退避重试，建议初始间隔 1 秒，最多重试 3 次。

## 核心接口列表

`POST /conversations` 创建会话，返回 conversation_id。

`POST /conversations/{id}/messages` 发送消息，支持同步返回机器人回答或异步回调两种模式。

`GET /conversations/{id}` 查询会话详情与完整消息历史。

`POST /tickets` 创建工单，需要提供 skill_group 字段指定技能组。

`GET /knowledge/search` 检索知识库，参数 q 为查询词，top_k 控制返回条数，默认 5，上限 20。

## 错误码约定

接口统一返回如下结构：`{"code": "OK", "message": "", "data": {}}`。

常见业务错误码：`INVALID_PARAM` 参数校验不通过；`UNAUTHORIZED` token 无效或过期；`FORBIDDEN` 当前账号无权访问该资源；`NOT_FOUND` 资源不存在；`RATE_LIMITED` 触发限流；`INTERNAL_ERROR` 服务端异常，需要联系技术支持。
