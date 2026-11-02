import { z } from "zod";
import { AIMessage, ToolMessage, tool } from "langchain";
import { createDeepAgent } from "deepagents";

import { model } from "../models.js";

const rollDice = tool(
  ({ sides }) => Math.floor(Math.random() * sides) + 1,
  {
    name: "roll_dice",
    description: "Roll a die with the given number of sides and return the result.",
    schema: z.object({ sides: z.number().int().default(6) }),
  },
);

const agent = createDeepAgent({
  model,
  tools: [rollDice],
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "Roll a 20-sided die for me." }],
});

for (const message of result.messages) {
  if (AIMessage.isInstance(message)) {
    for (const call of message.tool_calls ?? []) {
      console.log(`Model asked for: ${call.name}(${JSON.stringify(call.args)})`);
    }
  }
  if (ToolMessage.isInstance(message)) console.log(`Tool returned:   ${message.content}`);
}
console.log(`Final answer:    ${result.messages.at(-1)?.text}`);
