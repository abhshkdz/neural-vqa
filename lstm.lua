local LSTM = {}
function LSTM.create(input_size, rnn_size, num_layers)

    local inputs = {}

    table.insert(inputs, nn.Identity()()) -- x

    for L = 1, num_layers do
        table.insert(inputs, nn.Identity()()) -- c
        table.insert(inputs, nn.Identity()()) -- h
    end

    local x, input_size_L
    local outputs = {}

    for L = 1, num_layers do
        local prev_c = inputs[2*L]
        local prev_h = inputs[2*L + 1]

        if L == 1 then
            x = inputs[1]
            input_size_L = input_size
        else
            x = outputs[2*(L-1)]
            input_size_L = rnn_size
        end

        local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)
        local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
        local preactivations = nn.CAddTable()({i2h, h2h})

        -- gates
        local pre_sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(preactivations)
        local all_gates = nn.Sigmoid()(pre_sigmoid_chunk)

        -- input
        local in_chunk = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(preactivations)
        local in_transform = nn.Tanh()(in_chunk)

        local in_gate = nn.Narrow(2, 1, rnn_size)(all_gates)
        local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(all_gates)
        local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(all_gates)

        local next_c = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),
            nn.CMulTable()({in_gate, in_transform})
        })

        local c_transform = nn.Tanh()(next_c)
        local next_h = nn.CMulTable()({out_gate, c_transform})

        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end

    return nn.gModule(inputs, outputs)
end

return LSTM