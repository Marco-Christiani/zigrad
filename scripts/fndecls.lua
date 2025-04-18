--- run with `nvim --headless -c 'luafile scripts/fndecls.lua' -c 'q' build.zig`
---@diagnostic disable: unused-local
local function function_decls(path)
	local buf = vim.api.nvim_create_buf(false, true) -- scratch, unlisted
	local content = vim.fn.readfile(path)
	vim.api.nvim_buf_set_lines(buf, 0, -1, false, content)
	vim.api.nvim_set_option_value("filetype", "zig", { buf = buf })
	local parser = vim.treesitter.get_parser(buf)
	local tree = parser:parse()[1]
	local root = tree:root()

	-- Query fn decls
	local query = vim.treesitter.query.parse(
		"zig",
		[[
    ; Functions
    (function_declaration) @function
    ; (method_declaration) @function
    ; (arrow_function) @function
  ]]
	)
	local result = {}
	for id, node in query:iter_captures(root, buf, 0, -1) do
		local start_row, start_col, end_row, end_col = node:range()
		local lines = vim.api.nvim_buf_get_lines(buf, start_row, start_row + 1, false)
		if #lines > 0 then
			-- print(string.format("Line %d: %s", start_row + 1, lines[1]:gsub("^%s+", "")))
			local line = lines
				[1] -- stylua has bad taste for this tbh.
				:gsub("pub", "")
				:gsub("fn", "")
				:gsub("inline", "")
				:gsub("^%s*", "") -- remove leading space
				:gsub("[(].*[)].*", "") -- remove the rest of the line (eg. func(..).. becomes func)
			if string.find(line, "extern", 1, true) then -- skip extern
				goto continue
			end
			local currstr = string.format("%s", line)
			table.insert(result, currstr)
		end
		::continue::
	end
	return result
end

_G.FunctionDecls = function_decls -- for neovim

io.stdout:write("fn_list = [\n")
for _, file in ipairs(vim.fn.glob("**/*.zig", true, true)) do
	local decls = function_decls(file)
	if not decls or #decls == 0 then
		goto continue
	end
	io.stdout:write("\t# {{{ " .. file .. " " .. string.rep("-", 95 - #file) .. "\n")
	for _, fn in ipairs(decls) do
		io.stdout:write('\t"' .. fn .. '"' .. ",\n")
	end
	io.stdout:write("\t# }}}\n")
	::continue::
end
io.stdout:write("]")
