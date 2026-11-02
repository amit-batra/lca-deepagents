import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { createDeepAgent } from "deepagents";

import { model } from "../models.js";

const SERVER = join(dirname(fileURLToPath(import.meta.url)), "m1.6_dice_server.ts");

const client = new MultiServerMCPClient({
  dice: {
    transport: "stdio",
    command: process.execPath, // the Node running this script
    args: ["--import", "tsx", SERVER],
  },
});

try {
  const tools = await client.getTools(); // starts the server and asks which tools it has
  console.log("Tools from the server:", tools.map((t) => t.name));

  const agent = createDeepAgent({ model, tools });
  const result = await agent.invoke({
    messages: [{ role: "user", content: "Roll a 20-sided die for me." }],
  });
  console.log("Final answer:", result.messages.at(-1)?.text);
} finally {
  await client.close();
}
