"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import re
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def query(self, user_input: str) -> dict:
        # Trả lời 1 lượt, không gọi tool / không kết nối CSDL
        return {
            "status": "success",
            "answer": (
                "[Chatbot Baseline] Tôi không có kết nối cơ sở dữ liệu nên không thể "
                f"tra cứu chuyến bay hay thời tiết thực tế. Câu hỏi: {user_input}"
            ),
            "tool_calls": [],
        }


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    def run(self, user_input: str) -> dict:
        self.trace = []
        planned_actions = self._plan_actions(user_input)
        pending_actions = list(planned_actions)
        observations = []
        iteration = 0
        error_count = 0

        # Câu hỏi FAQ: không cần tool, trả Final Answer ngay
        if not planned_actions:
            answer = self._faq_answer(user_input)
            self.trace.append({
                "iteration": 1,
                "thought": "Đây là câu hỏi chính sách/FAQ, không cần gọi tool.",
                "action": None,
                "observation": None,
                "final_answer": answer,
            })
            return {
                "status": "completed",
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
            }

        while True:
            if iteration >= self.max_iterations:
                return {
                    "status": "max_iterations_reached",
                    "answer": "Không thể hoàn thành trong số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration,
                }

            iteration += 1

            if pending_actions:
                raw_action = pending_actions.pop(0)
                thought = f"Cần gọi tool {raw_action.get('name')} để lấy dữ liệu."
                action, observation = self._execute_action(raw_action)

                if isinstance(observation, dict) and observation.get("error"):
                    error_count += 1

                observations.append({
                    "tool": str(action.get("name", "")).strip().lower() if isinstance(action, dict) else "",
                    "result": observation,
                })
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "observation": observation,
                })

                # Trap 3: gặp lỗi 2 lần thì dừng và báo khách hàng
                if error_count >= 2:
                    answer = (
                        "Xin lỗi, hệ thống gặp lỗi khi tra cứu dữ liệu. "
                        "Quý khách vui lòng thử lại sau."
                    )
                    return {
                        "status": "completed",
                        "answer": answer,
                        "trace": self.trace,
                        "iterations": iteration,
                    }

                # Một tool duy nhất: đưa Final Answer ngay trong cùng iteration
                if not pending_actions and len(planned_actions) == 1:
                    answer = self._compose_answer(observations)
                    return {
                        "status": "completed",
                        "answer": answer,
                        "trace": self.trace,
                        "iterations": iteration,
                    }
            else:
                # Đa bước: iteration riêng cho Final Answer
                answer = self._compose_answer(observations)
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Đã đủ dữ liệu từ các tool, đưa ra Final Answer.",
                    "action": None,
                    "observation": None,
                    "final_answer": answer,
                })
                return {
                    "status": "completed",
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                }

    def _execute_action(self, raw_action):
        """Parse Action JSON an toàn rồi gọi tool trong TOOL_MAP."""
        try:
            if isinstance(raw_action, str):
                action = json.loads(raw_action)
            else:
                action = json.loads(json.dumps(raw_action))
        except (json.JSONDecodeError, TypeError, ValueError):
            return raw_action, "Invalid JSON format"

        if not isinstance(action, dict) or "name" not in action:
            return action, "Invalid JSON format"

        tool_name = str(action.get("name", "")).strip().lower()
        args = action.get("args") or {}

        if tool_name not in TOOL_MAP:
            return action, {"error": f"Unknown tool: {tool_name}"}

        try:
            observation = TOOL_MAP[tool_name](**args)
        except Exception as exc:
            observation = {"error": str(exc)}

        return action, observation

    def _plan_actions(self, user_input: str):
        text_lower = user_input.lower()

        is_faq = (
            "vinpearl" in text_lower
            or "chính sách" in text_lower
            or "đổi trả" in text_lower
        )
        has_weather = any(kw in text_lower for kw in ("thời tiết", "mặc gì"))
        has_flight = any(
            kw in text_lower
            for kw in ("chuyến bay", "tìm vé", "bay từ", "vé máy bay")
        ) and not is_faq

        actions = []

        if has_flight:
            origin, destination = self._parse_route(user_input)
            if origin and destination:
                actions.append({
                    "name": "get_flight_info",
                    "args": {
                        "origin": origin,
                        "destination": destination,
                        "max_price": self._parse_price(user_input),
                    },
                })

        if has_weather:
            city_code = self._parse_city_code(user_input)
            if city_code:
                actions.append({
                    "name": "get_weather_forecast",
                    "args": {"city_code": city_code},
                })

        return actions

    def _parse_route(self, text: str):
        match = re.search(
            r"từ\s+([A-Za-z]{3})\s+đi\s+([A-Za-z]{3})",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).upper(), match.group(2).upper()

        match = re.search(
            r"([A-Za-z]{3})\s+đến\s+([A-Za-z]{3})",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).upper(), match.group(2).upper()

        return None, None

    def _parse_city_code(self, text: str):
        weather_idx = text.lower().find("thời tiết")
        search_space = text[weather_idx:] if weather_idx >= 0 else text
        codes = re.findall(r"\b(HAN|SGN|DAD)\b", search_space.upper())
        if codes:
            return codes[0]

        upper = text.upper()
        if "ĐÀ NẴNG" in upper or "DA NANG" in upper:
            return "DAD"
        if "HỒ CHÍ MINH" in upper or "SÀI GÒN" in upper or "SAIGON" in upper:
            return "SGN"
        if "HÀ NỘI" in upper or "HA NOI" in upper:
            return "HAN"
        return None

    def _parse_price(self, text: str) -> int:
        normalized = text.lower().replace(",", ".")
        match = re.search(r"(\d+(?:\.\d+)?)\s*triệu", normalized)
        if match:
            return int(float(match.group(1)) * 1_000_000)

        match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", normalized)
        if match:
            return int(float(match.group(1)) * 1_000)

        return 5_000_000

    def _faq_answer(self, user_input: str) -> str:
        return (
            "Chính sách đổi trả vé máy bay Vinpearl: quý khách được đổi/trả vé "
            "theo điều kiện từng hạng vé. Vui lòng liên hệ tổng đài Vinpearl "
            "để được hỗ trợ chi tiết."
        )

    def _compose_answer(self, observations) -> str:
        parts = []
        for item in observations:
            result = item["result"]
            if item["tool"] == "get_flight_info":
                if not result:
                    parts.append("Không tìm thấy chuyến bay phù hợp với yêu cầu.")
                else:
                    flights = ", ".join(
                        f"{fl['flight_number']} ({fl['airline']}, "
                        f"{fl['price_vnd']} VND, {fl['departure_time']})"
                        for fl in result
                    )
                    parts.append(f"Các chuyến bay phù hợp: {flights}.")
            elif item["tool"] == "get_weather_forecast":
                if isinstance(result, dict) and result.get("error"):
                    parts.append(str(result["error"]))
                elif isinstance(result, dict):
                    parts.append(
                        f"Thời tiết {result.get('city')}: {result.get('temperature_c')}°C, "
                        f"{result.get('condition')}. Gợi ý: {result.get('recommendation')}"
                    )
            elif result == "Invalid JSON format":
                parts.append("Observation: Invalid JSON format")
        return " ".join(parts) if parts else "Không có dữ liệu để trả lời."


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
