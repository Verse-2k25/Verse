import { Geist, Geist_Mono } from "next/font/google";
import { ClerkProvider } from "@clerk/nextjs";

import "./globals.css";
import Link from "next/link"
import Navbar from "@/components/Navbar";
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  adjustFontFallback: true,
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  adjustFontFallback: true,
});

export const metadata = {
  title: "Verse",
  description: "Listen together, chat together.",
};

export default function RootLayout({ children }) {
  return (
    <ClerkProvider>
      <html lang="en">
      
        <body
          className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#252422] text-white`}
        >
        <Navbar/>
          <main className="min-h-screen ">
            {children}
          </main>
          {/* <footer className="p-4 border-t border-gray-800 text-center text-sm text-gray-400">
            &copy; {new Date().getFullYear()} Verse. All rights reserved.
          </footer> */}

        </body>
      </html>
    </ClerkProvider>
  );
}

