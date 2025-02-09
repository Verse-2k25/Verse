"use client"

import { Search, Menu } from "lucide-react";
import { SignInButton, SignOutButton, SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import Link from "next/link";
import { useEffect } from "react";
import axios from "axios";
import Cookies from "js-cookie";
import { getClerkId } from "@/lib/utils";

export default function Navbar() {
   

    useEffect(() => {
        const getUserData = async() => {
            const res = await axios.get(`/api/user`)
            Cookies.set('clerkId', res.data.user.id)
        }
        getUserData()
    },[])

    const userId = getClerkId();
    console.log("userId", userId)

    return (
        <nav className="bg-[#41362A]">
            {/* Upper Navbar */}
            <div className="px-3 py-3 border-b border-[#41362A]">
                <div className="flex items-center justify-between max-w-5xl mx-auto">
                    {/* Left - Logo/Image */}
                    <div className="flex items-center">
                        <img
                            src="/logo-main.png"
                            alt="Logo"
                            className="w-12 h-15"
                        />
                    </div>


                    <div className="text-[#E8DED4] text-3xl font-bold tracking-widest">
                        VERSE
                    </div>

                    <div className="flex items-center space-x-4">
                        <Link href="/Selection/election">
                            <button className="text-[#E8DED4] hover:text-white p-2">
                                <Search className="w-5 h-5" />
                            </button>
                        </Link>
                        <SignedOut>
                            <SignInButton>
                                <button className="text-[#E8DED4] hover:text-white text-sm font-medium">
                                    SIGN IN
                                </button>
                            </SignInButton>
                            <span className="text-[#3D261C] mx-2">|</span>
                            <Link href="/sign-up" className="text-[#E8DED4] hover:text-white text-sm font-medium">
                                JOIN
                            </Link>
                        </SignedOut>

                        <SignedIn>
                            <div className="flex items-center space-x-4">
                                <UserButton afterSignOutUrl="/Selection/election" />
                                <SignOutButton>
                                    <button className="text-[#E8DED4] hover:text-white text-sm font-medium">
                                        LOG OUT
                                    </button>
                                </SignOutButton>
                            </div>
                        </SignedIn>
                    </div>
                </div>
            </div>

            {/* Lower Navbar */}
            <div className="px-6 py-3 border-b border-[#3D261C]">
                <div className="max-w-7xl mx-auto flex items-center justify-between">
                    {/* Navigation Links */}
                    <div className="hidden md:flex items-center justify-center flex-1 space-x-12">
                        <Link href="/" className="text-[#E8DED4] hover:text-white text-sm font-medium">
                            HOME
                        </Link>
                        <Link href="/" className="text-[#E8DED4] hover:text-white text-sm font-medium">
                            FEATURES
                        </Link>
                        <Link href="/tune-in" className="text-[#E8DED4] hover:text-white text-sm font-medium">
                            TUNE IN
                        </Link>
                        <Link href="/recommendation" className="text-[#E8DED4] hover:text-white text-sm font-medium">
                            HANG OUT
                        </Link>
                        <Link href="/dashboard" className="text-[#E8DED4] hover:text-white text-sm font-medium">
                            DASHBOARD
                        </Link>
                    </div>

                    {/* Hamburger Menu */}
                    <button className="md:hidden text-[#E8DED4] hover:text-white p-2">
                        <Menu className="w-6 h-6" />
                    </button>
                </div>
            </div>
        </nav>
    );
}