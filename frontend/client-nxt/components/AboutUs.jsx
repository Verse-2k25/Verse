import { MessageCircle, Music2, Radio, Search, Users } from "lucide-react"
import {motion} from "framer-motion";
import { ChevronDown, ChevronUp } from "lucide-react";
import Image from "next/image"
import Link from "next/link"


export default function AboutUs(){

    return(

      <section className="bg-[#41362A] py-16 text-center ">
     <Music2 className="w-32 h-32 text-[#D77E1D] mx-auto mb-8" />
     <div className="max-w-2xl mx-auto">
       <h3 className="text-3xl font-bold text-white mb-4">Welcome to Verse!</h3>
       <p className="text-gray-400 leading-relaxed mb-8">
         Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin aliquet hendrerit sapien. In vel maximus
         ipsum. Ut eget magna at sem rhoncus ut. Cras tempor lorem quis nisi tincidunt, in volutpat tortor varius.
       </p>
       <button className="parallelogram-button bg-transparent border-2 border-[#D77E1D] text-[#D77E1D] font-bold py-4 px-16 hover:bg-[#D77E1D] hover:text-white transition-all uppercase tracking-wider">
         <span>JOIN US</span>
       </button>
     </div>
     </section>
    );
}