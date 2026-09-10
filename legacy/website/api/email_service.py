"""
AWS SES Email Service for NeuroShard Waitlist
Sends terminal-style confirmation emails to make users feel like they're
registering hardware for a mining operation.
"""

import os
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# AWS SES Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "noreply@neuroshard.com")
WEBSITE_URL = os.getenv("WEBSITE_URL", "https://neuroshard.com")


def get_ses_client():
    """Get AWS SES client with credentials from environment."""
    return boto3.client(
        'ses',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )


def generate_terminal_email_html(
    subject_line: str,
    log_entries: list[dict],
    footer_text: str = ""
) -> str:
    """
    Generate a terminal-style HTML email that looks like system output.
    Clean design with cyan accent color.
    """
    
    log_html = ""
    for entry in log_entries:
        level = entry.get("level", "INFO")
        level_color = {
            "INFO": "#94a3b8",      # slate-400
            "SUCCESS": "#22d3ee",   # cyan-400
            "WARN": "#fbbf24",      # amber-400
            "ERROR": "#f87171",     # red-400
            "SYSTEM": "#64748b",    # slate-500
            "DATA": "#22d3ee",      # cyan-400
        }.get(level, "#94a3b8")
        
        log_html += f'''
        <tr>
            <td style="color: #475569; font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace; font-size: 12px; padding: 2px 0; white-space: nowrap;">{entry.get("timestamp", "")}</td>
            <td style="color: {level_color}; font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace; font-size: 12px; padding: 2px 8px;">{entry.get("message", "")}</td>
        </tr>
        '''
    
    return f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: #020617; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <div style="max-width: 560px; margin: 0 auto; padding: 40px 20px;">
        
        <!-- Header -->
        <div style="text-align: center; margin-bottom: 40px;">
            <div style="font-size: 28px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">NeuroShard</div>
            <div style="font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px;">
                Distributed Neural Network
            </div>
        </div>
        
        <!-- Main Card -->
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; overflow: hidden;">
            
            <!-- Terminal Header -->
            <div style="background: #1e293b; padding: 12px 16px; display: flex; align-items: center;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background: #ef4444; margin-right: 6px;"></div>
                <div style="width: 10px; height: 10px; border-radius: 50%; background: #eab308; margin-right: 6px;"></div>
                <div style="width: 10px; height: 10px; border-radius: 50%; background: #22c55e; margin-right: 6px;"></div>
                <span style="color: #64748b; font-size: 12px; margin-left: 8px; font-family: 'SF Mono', monospace;">terminal</span>
            </div>
            
            <!-- Terminal Content -->
            <div style="padding: 20px;">
                <div style="color: #22d3ee; font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace; font-size: 13px; margin-bottom: 16px;">
                    $ neuroshard register --network mainnet
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    {log_html}
                </table>
            </div>
            
            <!-- Status Banner -->
            <div style="background: #0891b2; padding: 16px 20px;">
                <div style="color: #ffffff; font-size: 15px; font-weight: 600;">
                    {subject_line}
                </div>
            </div>
        </div>
        
        <!-- Footer Content -->
        <div style="margin-top: 24px;">
            {footer_text}
        </div>
        
        <!-- Footer Links -->
        <div style="text-align: center; padding: 30px 0; color: #475569; font-size: 12px;">
            <a href="{WEBSITE_URL}" style="color: #0891b2; text-decoration: none;">neuroshard.com</a>
            <span style="margin: 0 8px; color: #334155;">·</span>
            <a href="{WEBSITE_URL}/legal" style="color: #475569; text-decoration: none;">Terms</a>
            <div style="margin-top: 12px; color: #334155;">
                © 2025 NeuroShard Protocol
            </div>
        </div>
    </div>
</body>
</html>
'''


def send_waitlist_confirmation_email(
    to_email: str,
    referral_code: str,
    position: int,
    hardware_tier: str,
    hardware_score: int,
    estimated_daily_neuro: float,
    gpu_model: Optional[str] = None,
    ram_gb: int = 0,
    internet_speed: Optional[int] = None,
) -> bool:
    """
    Send waitlist confirmation email with terminal-style output.
    Makes the user feel like they just registered mining hardware.
    """
    
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    
    # Build log entries - clean and minimal
    log_entries = [
        {"timestamp": timestamp, "level": "SYSTEM", "message": "Initializing registration..."},
        {"timestamp": timestamp, "level": "INFO", "message": f"Email: {to_email}"},
    ]
    
    # Hardware details
    if gpu_model and gpu_model.lower() not in ["none", "cpu only", "no gpu"]:
        log_entries.append({"timestamp": timestamp, "level": "DATA", "message": f"GPU: {gpu_model}"})
    else:
        log_entries.append({"timestamp": timestamp, "level": "INFO", "message": "Mode: CPU compute"})
    
    log_entries.append({"timestamp": timestamp, "level": "DATA", "message": f"RAM: {ram_gb} GB"})
    
    if internet_speed:
        log_entries.append({"timestamp": timestamp, "level": "DATA", "message": f"Network: {internet_speed} Mbps"})
    
    log_entries.extend([
        {"timestamp": timestamp, "level": "SUCCESS", "message": f"Est. daily: {estimated_daily_neuro:.2f} NEURO"},
        {"timestamp": timestamp, "level": "SUCCESS", "message": "✓ Registration complete"},
    ])
    
    referral_url = f"{WEBSITE_URL}/join?ref={referral_code}"
    
    footer_text = f'''
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
            <div style="color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Your Referral Link</div>
            <div style="color: #22d3ee; font-family: 'SF Mono', monospace; font-size: 13px; word-break: break-all;">
                {referral_url}
            </div>
            <div style="color: #475569; font-size: 12px; margin-top: 10px;">
                Share to move up in the queue. +10 priority per referral.
            </div>
        </div>
        <p style="color: #94a3b8; font-size: 13px; line-height: 1.6; margin: 0;">
            Your hardware has been registered. We'll email you when your node is approved to join the network.
        </p>
    '''
    
    html_body = generate_terminal_email_html(
        subject_line="✓ Hardware Registered",
        log_entries=log_entries,
        footer_text=footer_text
    )
    
    # Plain text fallback
    text_body = f"""
NEUROSHARD - REGISTRATION CONFIRMED
===================================

Email: {to_email}
Estimated Daily: {estimated_daily_neuro:.2f} NEURO

Your Referral Link: {referral_url}

Share to move up in the queue. +10 priority per referral.

We'll email you when your node is approved.

- NeuroShard Protocol
{WEBSITE_URL}
"""
    
    return _send_email(
        to_email=to_email,
        subject="[NeuroShard] Registration Confirmed",
        html_body=html_body,
        text_body=text_body
    )


def send_waitlist_approval_email(
    to_email: str,
    referral_code: str,
) -> bool:
    """
    Send approval email when admin approves user to proceed with registration.
    """
    
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    
    log_entries = [
        {"timestamp": timestamp, "level": "SYSTEM", "message": "Authorization check..."},
        {"timestamp": timestamp, "level": "SUCCESS", "message": "✓ Node approved"},
        {"timestamp": timestamp, "level": "INFO", "message": f"Email: {to_email}"},
        {"timestamp": timestamp, "level": "SUCCESS", "message": "Ready to complete setup"},
    ]
    
    signup_url = f"{WEBSITE_URL}/signup?approved={referral_code}"
    
    footer_text = f'''
        <div style="text-align: center; margin: 24px 0;">
            <a href="{signup_url}" style="display: inline-block; background: #0891b2; color: white; padding: 14px 32px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 14px;">
                Complete Registration →
            </a>
        </div>
        <p style="color: #94a3b8; font-size: 13px; line-height: 1.6; margin: 0; text-align: center;">
            Create your account and wallet to start earning NEURO.
        </p>
    '''
    
    html_body = generate_terminal_email_html(
        subject_line="✓ Node Approved",
        log_entries=log_entries,
        footer_text=footer_text
    )
    
    text_body = f"""
NEUROSHARD - NODE APPROVED
==========================

Your hardware has been approved to join the NeuroShard network.

Complete your registration:
{signup_url}

Create your account and wallet to start earning NEURO.

- NeuroShard Protocol
{WEBSITE_URL}
"""
    
    return _send_email(
        to_email=to_email,
        subject="[NeuroShard] Your Node is Approved",
        html_body=html_body,
        text_body=text_body
    )


def _send_email(
    to_email: str,
    subject: str,
    html_body: str,
    text_body: str
) -> bool:
    """
    Internal function to send email via AWS SES.
    """
    
    # Check if AWS credentials are configured
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        logger.warning(f"AWS credentials not configured. Email to {to_email} not sent.")
        logger.info(f"Would have sent email with subject: {subject}")
        return False
    
    try:
        ses_client = get_ses_client()
        
        response = ses_client.send_email(
            Source=SENDER_EMAIL,
            Destination={
                'ToAddresses': [to_email]
            },
            Message={
                'Subject': {
                    'Data': subject,
                    'Charset': 'UTF-8'
                },
                'Body': {
                    'Text': {
                        'Data': text_body,
                        'Charset': 'UTF-8'
                    },
                    'Html': {
                        'Data': html_body,
                        'Charset': 'UTF-8'
                    }
                }
            }
        )
        
        logger.info(f"Email sent to {to_email}. Message ID: {response['MessageId']}")
        return True
        
    except ClientError as e:
        logger.error(f"Failed to send email to {to_email}: {e.response['Error']['Message']}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending email to {to_email}: {str(e)}")
        return False
