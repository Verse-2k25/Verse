
// const CLIENT_ID = "51a215f6113744d7bd6930816f384850"
// const CLIENT_SECRET = "0850771e9054447295bf0bf8d1c44b65"
"use client"

import { useState, useEffect } from "react"
import axios from "axios"
import { Card, CardContent } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"

const CLIENT_ID = "51a215f6113744d7bd6930816f384850"
const CLIENT_SECRET = "0850771e9054447295bf0bf8d1c44b65"

const genres = [
    "pop",
    "rock",
    "hip-hop",
    "jazz",
    "classical",
    "edm",
    "indie",
    "blues",
    "country",
    "metal",
    "reggae",
    "folk",
    "R&B",
    "soul",
    "funk",
    "techno",
    "house",
    "trance",
    "dubstep",
    "punk",
    "alternative",
    "grunge",
    "disco",
    "ska",
    "bluegrass",
    "gospel",
    "orchestra",
    "synthwave",
    "lo-fi",
    "k-pop",
    "j-pop",
    "c-pop",
    "latin",
    "afrobeats",
    "bollywood",
  ]

export default function Election({ userId }) {
  const [token, setToken] = useState("")
  const [artists, setArtists] = useState([])
  const [selectedGenres, setSelectedGenres] = useState([])
  const [selectedArtists, setSelectedArtists] = useState([])

  useEffect(() => {
    const getToken = async () => {
      try {
        const response = await axios.post(
          "https://accounts.spotify.com/api/token",
          new URLSearchParams({ grant_type: "client_credentials" }),
          {
            headers: {
              Authorization: `Basic ${btoa(`${CLIENT_ID}:${CLIENT_SECRET}`)}`,
              "Content-Type": "application/x-www-form-urlencoded",
            },
          },
        )
        setToken(response.data.access_token)
      } catch (error) {
        console.error("Error fetching token:", error)
      }
    }

    getToken()
  }, [])

  const fetchArtists = async () => {
    if (!token || selectedGenres.length === 0) return

    try {
      const genreQuery = selectedGenres.join(",")
      const response = await axios.get(`https://api.spotify.com/v1/search?q=genre:${genreQuery}&type=artist&limit=12`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      setArtists(response.data.artists.items)
    } catch (error) {
      console.error("Error fetching artists:", error)
    }
  }

  const handleGenreToggle = (genre) => {
    setSelectedGenres((prev) => (prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre]))
  }

  const handleArtistSelect = (artist) => {
    setSelectedArtists((prev) => (prev.includes(artist) ? prev.filter((a) => a !== artist) : [...prev, artist]))
  }

  const saveSelection = async () => {
    try {
      const response = await axios.post("/api/saveSelections", {
        userId, selectedGenres, selectedArtists
      })
      alert("Selection saved successfully!")
    } catch (error) {
      console.error("Error saving selection:", error)
    }
  }

  useEffect(() => {
    fetchArtists()
  }, [selectedGenres, token])

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#41362A] to-[#41362A] text-white p-5">
      <h1 className="text-4xl mt-8 font-bold text-center mb-8">Discover Artists by Genre</h1>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-center mb-6">Select Genres</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
          {genres.map((genre) => (
            <div key={genre} className="flex items-center space-x-2">
              <Checkbox id={genre} checked={selectedGenres.includes(genre)} onCheckedChange={() => handleGenreToggle(genre)}className="text-white border-white checked:bg-white checked:border-white"/>
              <label htmlFor={genre} className="text-sm font-medium">{genre.charAt(0).toUpperCase() + genre.slice(1)}</label>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {artists.map((artist) => (
          <Card key={artist.id} className="overflow-hidden bg-gray-800 text-white cursor-pointer" onClick={() => handleArtistSelect(artist.name)}>
            <CardContent className="p-0">
              <div className="relative aspect-square">
                <img src={artist.images[0]?.url || "/placeholder.svg"} alt={artist.name} className="object-cover w-full h-full" />
                <div className="absolute inset-0 bg-black bg-opacity-40 flex items-end p-4">
                  <div>
                    <h3 className="text-xl font-bold">{artist.name}</h3>
                    <p className="text-sm text-gray-300">Followers: {artist.followers.total.toLocaleString()}</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <button onClick={saveSelection} className="mt-8 mb-6 mx-auto block justify-center rounded-md border-2 border-[#D77E1D] text-white font-bold py-4 px-16 hover:bg-[#D77E1D] hover:text-white uppercase tracking-wider px-6 py-3 bg-[#D77E1D] hover:bg-transparent rounded-lg text-white  font-bold">Save Selection</button>
    </div>
  )
}
