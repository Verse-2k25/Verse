import mongoose from "mongoose"

const SelectionSchema = new mongoose.Schema({
  userId: { type: String, required: true },
  selectedGenres: { type: [String], required: true },
  selectedArtists: { type: [String], required: true },
}, { timestamps: true })

export default mongoose.models.Selection || mongoose.model("Selection", SelectionSchema)
