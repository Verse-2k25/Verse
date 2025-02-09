
import { MessageCircle, Music2, Radio, Search, Users } from "lucide-react";
import { SignInButton, SignOutButton, SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import Image from "next/image";
import Link from "next/link";
import AboutUs from "@/components/AboutUs";
import Features from "@/components/Features";
import HowIt from "@/components/HowIt";
import ContactUs from "@/components/ContactUs";
import Footer from "@/components/Footer";
import Chatbot from "@/components/chatbot/chatbot";
// import Election from "@/app/Selection/Election";
import Navbar from "@/components/Navbar";
import { currentUser } from "@clerk/nextjs/server";
import { cookies } from "next/headers";

export default async function Page() {

  // const user = await currentUser();
  // console.log(user)

  return (
    <div className="min-h-screen bg-[#EDE5D9]">


      {/* Hero Section */}
      <div>
        <section className="grid grid-cols-2 gap-8 p-16">
          <div className="space-y-6 pl-12">
            <h1 className="text-5xl font-bold text-[#5C4033]">
              Listen Together,
              <br />
              <span className="text-[#D77E1D]">Chat Together</span>
            </h1>
            <p className="text-gray-400 max-w-md">
              Create your own private music room, invite friends and enjoy synchronized music playback while chatting in
              real time.
            </p>
            <button className="parallelogram-button bg-[#D77E1D] text-white font-bold py-4 px-16 hover:bg-[#D77E2D] transition-all uppercase tracking-wider">
              <span>TRY OUT NOW</span>
            </button>
          </div>

          <div className="relative flex justify-center items-center">
            {SignedIn ? (
              <Image src="/hero.png" alt="Hero Image" width={500} height={500} className="transition-all duration-300" />
            ) : (
              <Image src="/hero.png" alt="Hero Image" width={500} height={500} />
            )}
          </div>
        </section>

        <div className="custom-shape-divider-top-1738904011">
          <svg data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 119" preserveAspectRatio="none">
            <path
              fill="#41362A"
              d="M985.66,92.83C906.67,72,823.78,31,743.84,14.19c-82.26-17.34-168.06-16.33-250.45.39-57.84,11.73-114,31.07-172,41.86A600.21,600.21,0,0,1,0,27.35V120H1200V95.8C1132.19,118.92,1055.71,111.31,985.66,92.83Z"
              className="shape-fill"
            ></path>
          </svg>
        </div>
      </div>

      <AboutUs />
      <Features />
      <Chatbot />
      {/* <Election  /> */}

      <HowIt />
      <ContactUs />
      <Footer />
    </div>
  );
}
