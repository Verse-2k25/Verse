"use client"

import { useState } from "react"
import axios from "axios"
import { motion } from "framer-motion"

export default function FriendRecommendations() {
  const [userId, setUserId] = useState("")
  const [recommendations, setRecommendations] = useState([])
  const [error, setError] = useState(null)


  
    try {
      const response = await axios.get(`http://localhost:5000/recommendations/${userId}`);
      setRecommendations(response.data);
      setError(null);
    } catch (error) {
      setError("Failed to fetch recommendations. Ensure the backend is running.");
    }
  };
  


  const convertToArray = (value) => {
    if (Array.isArray(value)) return value
    if (typeof value === "string") return value.split(/[,\s]+/).filter(Boolean)
    return []
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#EDE5D9] to-[#EDE5D9] p-8">
      <div className="">
        <h1 className="text-4xl font-bold mb-8 text-center text-[#D77E1D]">Friend Recommendations</h1>
        <div className="flex items-center space-x-4 mb-8 text-black">
          <input
            type="number"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Enter User ID"
            className="p-3 border-2 border-indigo-300 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200"
          />
          <button
            onClick={fetchRecommendations}
            className="bg-[#D77E1D]  text-white px-6 py-3 rounded-lg transition duration-200 transform hover:scale-105"
          >
            Get Recommendations
          </button>
        </div>
        {error && <p className="text-red-500 mb-6 text-center">{error}</p>}
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {recommendations.length > 0 ? (
            recommendations.map((friend) => (
              <motion.div
                key={friend.user_id}
                className="bg-white/80 backdrop-blur-sm shadow-lg rounded-2xl p-6 border border-indigo-100"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <h2 className="text-2xl font-semibold mb-4 text-indigo-800">{friend.name}</h2>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm font-medium text-gray-500 mb-2">Preferences</p>
                    <div className="flex flex-wrap gap-2">
                      {convertToArray(friend.preferences).map((pref, index) => (
                        <span
                          key={index}
                          className="px-4 py-1.5 rounded-full text-indigo-100 bg-indigo-900/80 backdrop-blur-sm text-sm font-medium"
                        >
                          {pref}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500 mb-2">Artists</p>
                    <div className="flex flex-wrap gap-2">
                      {convertToArray(friend.artists).map((artist, index) => (
                        <span
                          key={index}
                          className="px-4 py-1.5 rounded-full text-emerald-100 bg-emerald-900/80 backdrop-blur-sm text-sm font-medium"
                        >
                          {artist}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                <button className="w-full mt-6 bg-indigo-500 hover:bg-indigo-600 text-white py-2 rounded-lg transition duration-200 transform hover:scale-105">
                  Connect
                </button>
              </motion.div>
            ))
          ) : (
            <p className="text-gray-500 text-center col-span-full">No recommendations available.</p>
          )}
        </div>
      </div>
    </div>
  )
}

