import { MessageCircle, Music2, Radio, Search, Users } from "lucide-react";
import { motion } from "framer-motion";
import Image from "next/image";
import Link from "next/link";

export default function HowIt() {

  return (
    <section className="py-16 px-32 bg-[#EDE5D9]">
      <h2 className="text-3xl font-bold text-black text-center mb-14">
        How it Works
      </h2>
      <div className="grid grid-cols-3 gap-8">
        {[
          {
            number: "1",
            title: "Create room",
            description:
              "Sign up and create your music room. Customize settings and prepare for your friends.",
          },
          {
            number: "2",
            title: "Invite friends",
            description:
              "Generate an invite link and share it with your friends to join your music room.",
          },
          {
            number: "3",
            title: "Enjoy Together",
            description:
              "Start playing music and chat with your friends in real time. Create memories together",
          },
        ].map((step, index) => (
          <div key={index} className="relative pt-8 transition-all duration-500 ease-in-out transform hover:scale-105 hover:shadow-lg">
            <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-12 h-12 rounded-full bg-[#D77E1D] bg-opacity-60 flex items-center justify-center text-white font-bold text-xl">
              {step.number}

            </div>
            <div className="bg-[#41362A] bg-opacity-45 p-8 rounded-lg text-center min-h-[200px] flex flex-col items-center justify-center">
              <h3 className="text-white text-xl font-bold mb-4">{step.title}</h3>
              <p className="text-white font-medium text-sm">{step.description}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
