import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "dice", version: "1.0.0" });

server.registerTool(
  "roll_dice",
  {
    description: "Roll a die with the given number of sides and return the result.",
    inputSchema: { sides: z.number().int().default(6) },
  },
  // the same tool as Lesson 1.5
  async ({ sides }) => ({
    content: [{ type: "text", text: String(Math.floor(Math.random() * sides) + 1) }],
  }),
);

await server.connect(new StdioServerTransport()); // start the server and wait for a client
