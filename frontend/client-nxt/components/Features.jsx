import { MessageCircle, Music2, Radio, Search, Users } from "lucide-react";
import { motion } from "framer-motion";
import { ChevronDown, ChevronUp } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { CardBody, CardContainer, CardItem } from "../components/ui/3d-card";

export default function Features() {
  return (
    <section className="py-16 px-31">
      <h2 className="text-3xl font-bold text-black text-center mb-12">FEATURES</h2>
      <div className="grid grid-cols-3 gap-4">
        {[
          {
            icon: Music2,
            title: "Real-time music sync",
            description: "Synchronize music playback across all users in real-time",
          },
          {
            icon: MessageCircle,
            title: "Live Chat",
            description: "Chat with other listeners while enjoying music together",
          },
          { icon: Radio, title: "Admin control", description: "Powerful tools to manage your music room" },
          { icon: Users, title: "Invite System", description: "Invite friends to join your private music sessions" },
          { icon: Music2, title: "Song requests", description: "Let users request their favorite tracks" },
          { icon: Radio, title: "24/7 rooms", description: "Keep the music playing around the clock" },
        ].map((feature, index) => (
          <CardContainer key={index} className="inter-var">
            <CardBody className="bg-[#41362A] bg-opacity-70 p-8 rounded-xl text-center flex flex-col items-center justify-center w-auto sm:w-[25rem] h-auto space-y-4 dark:hover:shadow-2xl dark:hover:shadow-emerald-500/[0.1] dark:bg-black">
              <CardItem translateZ="100">
                <feature.icon className="w-12 h-12 text-[#D77E1D]" />
              </CardItem>
              <CardItem as="h3" translateZ="50" className="text-white font-semibold text-lg">
                {feature.title}
              </CardItem>
              <CardItem as="p" translateZ="60" className="text-gray-300 text-sm">
                {feature.description}
              </CardItem>
            </CardBody>
          </CardContainer>
        ))}
      </div>
    </section>
  );
}
