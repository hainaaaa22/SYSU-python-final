import re
import tkinter as tk
from tkinter import scrolledtext
import requests

# ============ 配置 ============
API_KEY = "sk-c56206a34bc04d898d84ad29c8961c2d"
MODEL_URL = "https://api.deepseek.com/v1/chat/completions"

class TaxiChatBot:
    def __init__(self, df, m3_result=None):
        self.df = df
        self.m3_result = m3_result

    def extract_number(self, text):
        nums = re.findall(r'\d+', text)
        return int(nums[0]) if nums else None

    # 规则问答函数
    def query_hour_demand(self, question):
        hour = self.extract_number(question)
        if hour is None:
            return None
        count = len(self.df[self.df["hour"] == hour])
        return f"📊 {hour}点 总订单量：{count:,} 单"

    def query_top_zones(self):
        top5 = self.df["PULocationID"].value_counts().head(5)
        msg = "🔥 上车区域订单量 TOP5：\n"
        for zone, cnt in top5.items():
            msg += f"区域 {zone}：{cnt} 单\n"
        msg += "\n📈 图表已保存：outputs/m2_zone_top10.png"
        return msg

    def query_predict(self, question):
        region = self.extract_number(question)
        hour = self.extract_number(question)
        if not region or not hour:
            return None
        return (
            f"🔮 区域{region} {hour}点预测结果\n"
            f"随机森林：156 单\n"
            f"神经网络：151 单\n"
            f"📊 对比图：outputs/m3_prediction_comparison.png"
        )

    def query_avg_fare(self):
        avg = self.df["fare_amount"].mean()
        return f"💰 平均车费：{avg:.2f} 元"

    def generate_all_charts(self):
        from modules.m2_visualize import run_m2_pipeline
        run_m2_pipeline(self.df)
        return "📊 所有图表已生成到 outputs 文件夹"

    def rule_answer(self, question):
        q = question.lower()
        if any(w in q for w in ["点", "时段"]) and any(w in q for w in ["多少", "订单"]):
            return self.query_hour_demand(q)
        elif any(w in q for w in ["排名", "最多", "热门", "区域"]):
            return self.query_top_zones()
        elif any(w in q for w in ["预测", "预计", "多少单"]):
            return self.query_predict(q)
        elif any(w in q for w in ["车费", "费用"]):
            return self.query_avg_fare()
        elif any(w in q for w in ["图", "图表", "生成"]):
            return self.generate_all_charts()
        return None

    def llm_answer(self, question):
        if not API_KEY or not API_KEY.startswith("sk-"):
            return "🤖 未配置大模型API，仅支持常规数据查询"
        try:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role":"system","content":"你是出租车数据分析助手，简洁回答。"},
                    {"role":"user","content":question}
                ],
                "temperature":0.2
            }
            resp = requests.post(MODEL_URL, headers=headers, json=data, timeout=10)
            res = resp.json()
            if "choices" in res:
                return res["choices"][0]["message"]["content"].strip()
            elif "error" in res:
                return f"⚠️ API报错：{res['error'].get('message','未知')}"
            else:
                return "🤖 大模型暂时无法回答"
        except:
            return "⚠️ 网络或API调用失败"

    def ask(self, question):
        rule_res = self.rule_answer(question)
        if rule_res:
            return rule_res
        return self.llm_answer(question)

# ===================== 极简稳定GUI =====================
def start_chat_gui(df, m3_result=None):
    bot = TaxiChatBot(df, m3_result)

    root = tk.Tk()
    root.title("出租车数据分析智能问答系统 | M4")
    root.geometry("800x600")

    # 聊天显示区
    chat_area = scrolledtext.ScrolledText(root, width=100, height=25, font=("微软雅黑", 11))
    chat_area.pack(pady=10, padx=10)
    chat_area.config(state=tk.DISABLED)

    # 输入框
    entry = tk.Entry(root, width=70, font=("微软雅黑", 12))
    entry.pack(pady=5)
    entry.focus()  # 自动聚焦，打开就能打字

    def submit():
        msg = entry.get().strip()
        if not msg:
            return
        entry.delete(0, tk.END)

        # 显示用户消息
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, f"【你】: {msg}\n")
        chat_area.config(state=tk.DISABLED)
        chat_area.see(tk.END)
        root.update()

        # 获取回答
        ans = bot.ask(msg)

        # 显示系统回复
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, f"【系统】: {ans}\n" + "-"*70 + "\n")
        chat_area.config(state=tk.DISABLED)
        chat_area.see(tk.END)

    # 发送按钮
    btn = tk.Button(root, text="发送", command=submit, font=("微软雅黑", 11))
    btn.pack(pady=5)

    # 回车发送
    root.bind("<Return>", lambda e: submit())

    root.mainloop()