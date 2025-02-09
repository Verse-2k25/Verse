"use client"
// Second file: components/ContactUs.jsx
import { MessageCircle, Music2, Radio, Search, Users } from "lucide-react";
import { motion } from "framer-motion";
import { ChevronDown, ChevronUp } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useState } from 'react';
import toast, { Toaster } from 'react-hot-toast';

export default function ContactUs() {
  const [status, setStatus] = useState({
    submitted: false,
    submitting: false,
    info: { error: false, msg: null }
  });

  const [inputs, setInputs] = useState({
    fromName: '',
    fromMail: '',
    message: ''
  });

  const handleChange = (e) => {
    setInputs(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));

    if (status.submitted) {
      setStatus({
        submitted: false,
        submitting: false,
        info: { error: false, msg: null }
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus(prevStatus => ({ ...prevStatus, submitting: true }));
    const loadingToast = toast.loading('Sending your message...');

    try {
      const mailData = {
        fromName: inputs.fromName,
        toName: 'Debaditya', // Recipient name
        subject: 'New Contact Form Submission',
        text: inputs.message,
        html: `
          <div style="font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5;">
            <h2 style="color: #333;">New Contact Form Submission</h2>
            <div style="background-color: white; padding: 20px; border-radius: 5px;">
              <p><strong>From:</strong> ${inputs.fromName}</p>
              <p><strong>Email:</strong> ${inputs.fromMail}</p>
              <p><strong>Message:</strong></p>
              <p style="white-space: pre-wrap;">${inputs.message}</p>
            </div>
          </div>
        `,
        fromMail: inputs.fromMail,
        toMail: 'debaditya.22@aot.edu.in'
      };

      const res = await fetch('/api/sendMail', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(mailData)
      });

      const data = await res.json();

      if (res.status === 200) {
        setStatus({
          submitted: true,
          submitting: false,
          info: { error: false, msg: 'Message sent successfully!' }
        });

        setInputs({
          fromName: '',
          fromMail: '',
          message: ''
        });

        toast.success('Message sent successfully!', {
          id: loadingToast,
          duration: 5000,
          icon: '✉️',
        });
      } else {
        throw new Error(data.message || 'Error sending message');
      }
    } catch (error) {
      console.error('Error:', error);
      setStatus({
        submitted: false,
        submitting: false,
        info: { error: true, msg: error.message }
      });

      toast.error(error.message || 'Failed to send message. Please try again.', {
        id: loadingToast,
        duration: 5000,
      });
    }
  };

  return (
    <section className="py-16 px-32 bg-[#41362A]">
      <Toaster 
        position="top-right"
        toastOptions={{
          style: {
            background: '#333',
            color: '#fff',
            border: '1px solid #D77E1D',
          },
          success: {
            style: {
              background: '#2b3c2b',
            },
          },
          error: {
            style: {
              background: '#3c2b2b',
            },
          },
        }}
      />
      
      <h2 className="text-2xl font-bold text-white text-center mb-12">FEEDBACK</h2>
      <div className="max-w-2xl mx-auto">
        <form className="space-y-6" onSubmit={handleSubmit}>
          <div>
            <label htmlFor="fromName" className="block text-sm font-medium text-white">
              Name
            </label>
            <input
              type="text"
              id="fromName"
              name="fromName"
              value={inputs.fromName}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md bg-[#EDE5D9] border-gray-700 bg-opacity-50 shadow-sm focus:border-[#1A1E24] focus:ring focus:ring-[#1A1E24] focus:ring-opacity-50"
              required
            />
          </div>
          
          <div>
            <label htmlFor="fromMail" className="block text-sm font-medium text-white">
              Email
            </label>
            <input
              type="email"
              id="fromMail"
              name="fromMail"
              value={inputs.fromMail}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md bg-[#EDE5D9] border-gray-700 bg-opacity-50 shadow-sm focus:border-[#4DB6AC] focus:ring focus:ring-[#4DB6AC] focus:ring-opacity-50"
              required
            />
          </div>
          
          <div>
            <label htmlFor="message" className="block text-sm font-medium text-white">
              Message
            </label>
            <textarea
              id="message"
              name="message"
              rows={4}
              value={inputs.message}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md bg-[#EDE5D9] border-gray-700 bg-opacity-50 shadow-sm focus:border-[#4DB6AC] focus:ring focus:ring-[#4DB6AC] focus:ring-opacity-50"
              required
            ></textarea>
          </div>
          
          <div>
            <button
              type="submit"
              disabled={status.submitting}
              className="w-full flex justify-center py-2 px-4 border border-[#D77E1D] rounded-md bg-transparent border-2 border-[#D77E1D] text-white font-bold py-4 px-16 hover:bg-[#D77E1D] hover:text-white transition-all uppercase tracking-wider disabled:opacity-50"
            >
              {status.submitting ? 'Sending...' : 'Send Message'}
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}