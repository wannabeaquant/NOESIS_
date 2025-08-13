import smtplib
from email.mime.text import MIMEText
import os

class EmailAlert:
    def __init__(self, smtp_server: str = None, smtp_port: int = None, username: str = None, password: str = None):
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.username = username or os.getenv("SMTP_USERNAME")
        self.password = password or os.getenv("SMTP_PASSWORD")
        
        # Check if email configuration is complete
        if not all([self.smtp_server, self.username, self.password]):
            print("Warning: Incomplete email configuration. Email alerts will be disabled.")
            self.enabled = False
        else:
            self.enabled = True

    def send_alert(self, to_email: str, subject: str, message: str):
        if not self.enabled:
            print(f"Email alert disabled. Would send to {to_email}: {subject}")
            return
            
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.username
            msg['To'] = to_email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.username, [to_email], msg.as_string())
            print(f"Email alert sent to {to_email}")
        except Exception as e:
            print(f"Error sending email alert: {e}") 