import { auth, currentUser } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

export async function GET(){
    const {userId} = await auth();

    if(!userId) {
        return new NextResponse('Unauthorized', {status: 401})
    }

    const user = await currentUser();

    return NextResponse.json({
        user: user
    })

}