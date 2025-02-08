export async function POST(request: Request) {
  // Parse the incoming JSON from the client
  const { messages } = await request.json();
  console.log("Incoming messages", messages);

  try {
    // Forward the request to the Python backend
    const backendResponse = await fetch("http://127.0.0.1:8000/get_response", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ messages }),
    });

    // Check if the backend response is OK (status 200-299)
    if (!backendResponse.ok) {
      throw new Error(`Backend error: ${backendResponse.statusText}`);
    }

    // Parse the JSON response from the backend
    const data = await backendResponse.json();

    // Return the backend's JSON response
    return new Response(JSON.stringify(data), {
      status: backendResponse.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error: any) {
    console.error("Error in POST handler:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}

