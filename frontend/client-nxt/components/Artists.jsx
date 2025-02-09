import { MessageCircle, Music2, Radio, Search, Users } from "lucide-react"
import {motion} from "framer-motion";
import { ChevronDown, ChevronUp } from "lucide-react";
import Image from "next/image"
import Link from "next/link"


export default function Artists(){

    return(
        <section className="py-16 px-32 bg-[#41362A] bg-opacity-85">
        <h2 className="text-2xl font-bold text-white text-center mb-12">TUNE IN NOW</h2>
        <div className="flex justify-between items-center">
          {[
            {
              name: "A.R Rahman",
              image: "https://i.scdn.co/image/ab6761610000e5ebb19af0ea736c6228d6eb539c", 
            },
            {
              name: "Bruno Mars",
              image: "https://i.scdn.co/image/ab6761610000e5ebc36dd9eb55fb0db4911f25dd", 
            },
            {
              name: "Coldplay",
              image: "https://t2.gstatic.com/images?q=tbn:ANd9GcSpiz6GsEsfY8DPUs41wZvj6eIryeHG4kKyKqIWTl3y20lnWKxr", 
            },
            {
              name: "Dua Lipa",
              image: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRsm65no-6Y65rB2Bo5FJ5jJ4R1JMyw0jmAA&s", 
            },
            {
              name: "Elvis Presley",
              image: "https://cdn.britannica.com/85/202285-050-EF215325/Elvis-Presley-Girl-Happy-Boris-Sagal.jpg", 
            },
          ].map((artist, index) => (
            <div key={index} className="text-center space-y-4">
              <div className="w-24 h-24 rounded-full overflow-hidden mx-auto">
                <img
                  src={artist.image}  
                  alt={artist.name}
                  width={100}
                  height={100}
                  className="object-cover"
                />
              </div>
              <p className="text-white text-sm">{artist.name}</p>
            </div>
          ))}
        </div>
      </section>
      

    );
}


