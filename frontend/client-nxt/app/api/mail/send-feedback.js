// First file: pages/api/sendMail.js
import nodemailer from 'nodemailer';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const { fromName, toName, subject, text, html, fromMail, toMail } = req.body;

    // Validate required fields
    if (!fromMail || !toMail || !subject || !toName || !fromName || !text || !html) {
      return res.status(400).json({ message: 'Please provide all the fields' });
    }

    // Create mail transporter
    const transporter = nodemailer.createTransport({
      service: 'gmail',
      auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS
      }
    });

    // Mail options
    const mailOptions = {
      from: `"${fromName}" <${fromMail}>`,
      to: `"${toName}" <${toMail}>`,
      subject: subject,
      text: text,
      html: html
    };

    // Send mail
    const info = await transporter.sendMail(mailOptions);
    
    if (!info) {
      return res.status(401).json({ message: 'Invalid Mail' });
    }

    res.status(200).json({
      message: 'Mail sent successfully',
      info: info
    });

  } catch (err) {
    console.error('Error sending mail:', err);
    res.status(500).json({ message: 'Error sending mail', error: err.message });
  }
}

