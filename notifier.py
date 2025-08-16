import smtplib
from email.mime.text import MIMEText
from typing import List, Dict


def send_email(signals: List[Dict[str, str]], user: str, password: str, to: str):
    body_lines = []
    for s in signals:
        rules_met = ', '.join([k for k, v in s['rules'].items() if v])
        body_lines.append(f"{s['ticker']}: {s['action']} (conf={s['confidence']:.2f}) rules: {rules_met}")
    body = '\n'.join(body_lines)
    msg = MIMEText(body)
    msg['Subject'] = 'Daily Trading Signals'
    msg['From'] = user
    msg['To'] = to
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(user, password)
        smtp.sendmail(user, [to], msg.as_string())
