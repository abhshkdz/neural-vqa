-- Messy but works.

local DataLoader = {}
DataLoader.__index = DataLoader

function DataLoader.create(data_dir, batch_size, opt)

    local self = {}
    setmetatable(self, DataLoader)

    local train_questions_file = path.join(data_dir, 'MultipleChoice_mscoco_train2014_questions.json')
    local train_annotations_file = path.join(data_dir, 'mscoco_train2014_annotations.json')

    local val_questions_file = path.join(data_dir, 'MultipleChoice_mscoco_val2014_questions.json')
    local val_annotations_file = path.join(data_dir, 'mscoco_val2014_annotations.json')

    local questions_vocab_file = path.join(data_dir, 'questions_vocab.t7')
    local answers_vocab_file = path.join(data_dir, 'answers_vocab.t7')

    local tensor_file = path.join(data_dir, 'data.t7')
    local embeddings_file = path.join(data_dir, 'q_200d_glove_embeddings.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing

    local run_prepro = false
    if not (path.exists(questions_vocab_file) and path.exists(answers_vocab_file) and path.exists(tensor_file)) then
        print('questions_vocab.t7, answers_vocab.t7 or data.t7 files do not exist. Running preprocessing...')
        run_prepro = true
    else
        local train_questions_attr = lfs.attributes(train_questions_file)
        local questions_vocab_attr = lfs.attributes(questions_vocab_file)
        local tensor_attr = lfs.attributes(tensor_file)

        if train_questions_attr.modification > questions_vocab_attr.modification or train_questions_attr.modification > tensor_attr.modification then
            print('questions_vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing...')
        DataLoader.json_to_tensor(train_questions_file, train_annotations_file, val_questions_file, val_annotations_file, questions_vocab_file, answers_vocab_file, tensor_file)
    end

    print('loading data files...')
    self.data = torch.load(tensor_file)
    self.q_vocab_mapping = torch.load(questions_vocab_file)
    self.a_vocab_mapping = torch.load(answers_vocab_file)
    self.q_embeddings = torch.load(embeddings_file)

    self.q_vocab_size = 0
    for _ in pairs(self.q_vocab_mapping) do
        self.q_vocab_size = self.q_vocab_size + 1
    end

    self.a_vocab_size = 0
    for _ in pairs(self.a_vocab_mapping) do
        self.a_vocab_size = self.a_vocab_size + 1
    end

    if opt.fc7_file ~= '0' then
        print('Loading fc7 features from ' .. opt.fc7_file)
        self.fc7 = torch.load(opt.fc7_file)
        self.fc7_image_id = torch.load(opt.fc7_image_id_file)
        self.fc7_mapping = {}
        for i, v in pairs(self.fc7_image_id) do
            self.fc7_mapping[v] = i
        end
    end

    self.batch_size = batch_size
    self.nbatches = 0
    self.counts = {}

    local idx = 1
    for i, v in pairs(self.data.q_train_count_idx_mapping) do
        if #v < self.batch_size then
            self.data.q_train_count_idx_mapping[i] = nil
        else
            self.counts[idx] = i
            idx = idx + 1
            if #v % self.batch_size ~= 0 then
                for j = (#v - #v % self.batch_size + 1), #v do
                    self.data.q_train_count_idx_mapping[i][j] = nil
                end
            end
            self.nbatches = self.nbatches + #self.data.q_train_count_idx_mapping[i] / self.batch_size
        end
    end

    self.batch_data = {['train'] = {}}
    self.batch_data.train = {
        ['question'] = {},
        ['answer'] = torch.ShortTensor(self.batch_size * self.nbatches),
        ['image_feat'] = torch.DoubleTensor(self.batch_size * self.nbatches, 4096)
    }

    if opt.gpuid >= 0 then
        self.batch_data.train.image_feat = self.batch_data.train.image_feat:cuda()
    end

    idx = 1
    for i, v in pairs(self.data.q_train_count_idx_mapping) do
        self.batch_data.train.question[i] = torch.ShortTensor(#v, i)
        for j = 1, #v do
            self.batch_data.train.question[i][j] = self.data.train[v[j]]['question']
            self.batch_data.train.answer[idx] = self.data.train[v[j]]['answer']
            self.batch_data.train.image_feat[idx] = self.fc7[self.fc7_mapping[self.data.train[v[j]]['image_id']]]
            idx = idx + 1
        end
        if opt.gpuid >= 0 then
            self.batch_data.train.question[i] = self.batch_data.train.question[i]:cuda()
        end
    end

    if opt.gpuid >= 0 then
        self.batch_data.train.answer = self.batch_data.train.answer:cuda()
    end

    self.data.train = nil

    self.batch_idx = 1
    self.count_idx = 1
    self.cnt_batch_idx = 1

    collectgarbage()
    return self

end

function DataLoader:next_batch()
    if self.batch_idx - 1 == self.nbatches then self.batch_idx = 1 self.cnt_batch_idx = 1 self.count_idx = 1 end

    if self.cnt_batch_idx * self.batch_size - 1 > self.batch_data.train.question[self.counts[self.count_idx]]:size(1) then
        self.count_idx = self.count_idx + 1
        self.cnt_batch_idx = 1
    end

    question = self.batch_data.train.question[self.counts[self.count_idx]]:narrow(1, (self.cnt_batch_idx - 1) * self.batch_size + 1, self.batch_size)
    answer = self.batch_data.train.answer:narrow(1, (self.batch_idx - 1) * self.batch_size + 1, self.batch_size)
    image = self.batch_data.train.image_feat:narrow(1, (self.batch_idx - 1) * self.batch_size + 1, self.batch_size)

    self.batch_idx = self.batch_idx + 1
    self.cnt_batch_idx = self.cnt_batch_idx + 1

    return question, answer, image
end

function DataLoader.json_to_tensor(in_train_q, in_train_a, in_val_q, in_val_a, out_vocab_q, out_vocab_a, out_tensor)

    local JSON = (loadfile "utils/JSON.lua")()

    print('creating vocabulary mapping...')

    -- build answer vocab using train+val

    local f = torch.DiskFile(in_train_a)
    c = f:readString('*a')
    local train_a = JSON:decode(c)
    f:close()

    f = torch.DiskFile(in_val_a)
    c = f:readString('*a')
    local val_a = JSON:decode(c)
    f:close()

    local unordered = {}

    for i = 1, #train_a['annotations'] do
        local token = train_a['annotations'][i]['multiple_choice_answer']
        if not unordered[token] then
            unordered[token] = 1
        else
            unordered[token] = unordered[token] + 1
        end
    end

    for i = 1, #val_a['annotations'] do
        local token = val_a['annotations'][i]['multiple_choice_answer']
        if not unordered[token] then
            unordered[token] = 1
        else
            unordered[token] = unordered[token] + 1
        end
    end

    local sorted_a = get_keys_sorted_by_value(unordered, function(a, b) return a > b end)

    local top_n = 1000
    local ordered = {}
    for i = 1, top_n do
        ordered[#ordered + 1] = sorted_a[i]
    end
    ordered[#ordered + 1] = "UNK"
    table.sort(ordered)

    local a_vocab_mapping = {}
    for i, word in ipairs(ordered) do
        a_vocab_mapping[word] = i
    end

    -- build question vocab using train+val

    f = torch.DiskFile(in_train_q)
    c = f:readString('*a')
    local train_q = JSON:decode(c)
    f:close()

    f = torch.DiskFile(in_val_q)
    c = f:readString('*a')
    local val_q = JSON:decode(c)
    f:close()

    unordered = {}

    local q_train_count_idx_mapping = {}
    local q_val_count_idx_mapping = {}

    for i = 1, #train_q['questions'] do
        local count = 0
        for token in word_iter(train_q['questions'][i]['question']) do
            if not unordered[token] then
                unordered[token] = 1
            else
                unordered[token] = unordered[token] + 1
            end
            count = count + 1
        end
        if a_vocab_mapping[train_a['annotations'][i]['multiple_choice_answer']] then
            if not q_train_count_idx_mapping[count] then q_train_count_idx_mapping[count] = {} end
            q_train_count_idx_mapping[count][#q_train_count_idx_mapping[count]+1] = i
        end
    end

    for i = 1, #val_q['questions'] do
        local count = 0
        for token in word_iter(val_q['questions'][i]['question']) do
            if not unordered[token] then
                unordered[token] = 1
            else
                unordered[token] = unordered[token] + 1
            end
            count = count + 1
        end
        if a_vocab_mapping[val_a['annotations'][i]['multiple_choice_answer']] then
            if not q_val_count_idx_mapping[count] then q_val_count_idx_mapping[count] = {} end
            q_val_count_idx_mapping[count][#q_val_count_idx_mapping[count]+1] = i
        end
    end

    local threshold = 0
    local ordered = {}
    for token, count in pairs(unordered) do
        if count > threshold then
            ordered[#ordered + 1] = token
        end
    end
    ordered[#ordered + 1] = "UNK"
    table.sort(ordered)

    local q_vocab_mapping = {}
    for i, word in ipairs(ordered) do
        q_vocab_mapping[word] = i
    end

    print('putting data into tensor...')

    -- save train+val data

    local data = {
        train = {},
        val = {},
        q_train_count_idx_mapping = q_train_count_idx_mapping,
        q_val_count_idx_mapping = q_val_count_idx_mapping
    }

    for i = 1, #train_q['questions'] do
        -- if a_vocab_mapping[train_a['annotations'][i]['multiple_choice_answer']] then
            local question = {}
            local wl = 0
            for token in word_iter(train_q['questions'][i]['question']) do
                wl = wl + 1
                question[wl] = q_vocab_mapping[token] or q_vocab_mapping["UNK"]
            end
            data.train[i] = {
                image_id = train_a['annotations'][i]['image_id'],
                question = torch.ShortTensor(wl),
                answer = a_vocab_mapping[train_a['annotations'][i]['multiple_choice_answer']]
            }
            for j = 1, wl do
                data.train[i]['question'][j] = question[j]
            end
        -- end
    end

    for i = 1, #val_q['questions'] do
        -- if a_vocab_mapping[val_a['annotations'][i]['multiple_choice_answer']] then
            local question = {}
            local wl = 0
            for token in word_iter(val_q['questions'][i]['question']) do
                wl = wl + 1
                question[wl] = q_vocab_mapping[token] or q_vocab_mapping["UNK"]
            end
            data.val[i] = {
                image_id = val_a['annotations'][i]['image_id'],
                question = torch.ShortTensor(wl),
                answer = a_vocab_mapping[val_a['annotations'][i]['multiple_choice_answer']]
            }
            for j = 1, wl do
                data.val[i]['question'][j] = question[j]
            end
        -- end
    end

    -- save output preprocessed files
    print('saving ' .. out_vocab_q)
    torch.save(out_vocab_q, q_vocab_mapping)
    print('saving ' .. out_vocab_a)
    torch.save(out_vocab_a, a_vocab_mapping)
    print('saving ' .. out_tensor)
    torch.save(out_tensor, data)

end

function word_iter(str)
    return string.gmatch(str, "%a+")
end

function get_keys_sorted_by_value(tbl, sort_fn)
    local keys = {}
    for key in pairs(tbl) do
        table.insert(keys, key)
    end

    table.sort(keys, function(a, b)
        return sort_fn(tbl[a], tbl[b])
    end)

    return keys
end

return DataLoader