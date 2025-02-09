import { clsx } from "clsx";
import Cookies from "js-cookie";
import { twMerge } from "tailwind-merge"

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}


export const getClerkId = () => {
  const id = Cookies.get('clerkId');
  return id
}