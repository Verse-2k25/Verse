import React from 'react';
import Image from 'next/image';

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-[#f5e6d3] p-6">
      <div className=" mx-auto grid grid-cols-1 md:grid-cols-12 gap-2">
        {/* Top Row */}
        <div className="md:col-span-3 flex justify-center items-center bg-[#e6d5bc] rounded-lg shadow-md p-3">
          <Image
            src="/most-streamed-tracks.png"
            alt="Most streamed tracks"
            width={400}
            height={300}
            className="w-full"
          />
        </div>
        
        <div className="md:col-span-6 bg-[#e6d5bc] rounded-lg shadow-md p-4">
          <Image
            src="/streams-chart.png"
            alt="Streams chart"
            width={800}
            height={300}
            className="w-full h-auto"
          />
        </div>

        {/* Statistics Cards */}
        <div className="md:col-span-3 flex flex-col gap-4">
          <div className="bg-[#e6d5bc] rounded-lg shadow-md p-4">
            <Image
              src="/total-streams.png"
              alt="Total streams"
              width={200}
              height={100}
              className="w-full h-auto"
            />
          </div>
          
          <div className="bg-[#e6d5bc] rounded-lg shadow-md p-4">
            <Image
              src="/artist-count.png"
              alt="Artist count"
              width={200}
              height={100}
              className="w-full h-auto"
            />
          </div>
        </div>

        {/* Bottom Row */}
        <div className="md:col-span-7 bg-[#e6d5bc] rounded-lg shadow-md p-4">
          <Image
            src="/artist-stats.png"
            alt="Artist statistics"
            width={800}
            height={300}
            className="w-full h-auto"
          />
        </div>
        
        <div className="md:col-span-5 bg-[#e6d5bc] rounded-lg shadow-md p-4">
          <Image
            src="/artist-distribution.png"
            alt="Artist distribution"
            width={400}
            height={300}
            className="w-full h-auto"
          />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;