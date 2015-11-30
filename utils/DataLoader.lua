-- Messy but works.

local DataLoader = {}
DataLoader.__index = DataLoader

function DataLoader.create(data_dir, batch_size, opt, mode)

    local self = {}
    setmetatable(self, DataLoader)

    self.mode = mode or 'train'

    local train_questions_file = path.join(data_dir, 'MultipleChoice_mscoco_train2014_questions.json')
    local train_annotations_file = path.join(data_dir, 'mscoco_train2014_annotations.json')

    local val_questions_file = path.join(data_dir, 'MultipleChoice_mscoco_val2014_questions.json')
    local val_annotations_file = path.join(data_dir, 'mscoco_val2014_annotations.json')

    local questions_vocab_file = path.join(data_dir, 'questions_vocab.t7')
    local answers_vocab_file = path.join(data_dir, 'answers_vocab.t7')

    local tensor_file = path.join(data_dir, 'data.t7')

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

    print('Loading data files...')
    local data = torch.load(tensor_file)
    if mode == 'fc7_feat' then
        self.data = data
        collectgarbage()
        return self
    end

    self.q_max_length = data.q_max_length
    self.q_vocab_mapping = torch.load(questions_vocab_file)
    self.a_vocab_mapping = torch.load(answers_vocab_file)

    self.q_vocab_size = 0
    for _ in pairs(self.q_vocab_mapping) do
        self.q_vocab_size = self.q_vocab_size + 1
    end

    self.a_vocab_size = 0
    for _ in pairs(self.a_vocab_mapping) do
        self.a_vocab_size = self.a_vocab_size + 1
    end

    self.batch_size = batch_size

    if mode == 'predict' then
        collectgarbage()
        return self
    end

    self.train_nbatches = 0
    self.val_nbatches = 0

    -- Load train into batches

    print('Loading train fc7 features from ' .. opt.train_fc7_file)
    local fc7 = torch.load(opt.train_fc7_file)
    local fc7_image_id = torch.load(opt.train_fc7_image_id_file)
    local fc7_mapping = {}
    for i, v in pairs(fc7_image_id) do
        fc7_mapping[v] = i
    end

    self.batch_data = {['train'] = {}, ['val'] = {}}

    self.batch_data.train = {
        ['question'] = torch.ShortTensor(self.batch_size * math.floor(#data.train / self.batch_size), data.q_max_length),
        ['answer'] = torch.ShortTensor(self.batch_size * math.floor(#data.train / self.batch_size)),
        ['image_feat'] = torch.DoubleTensor(self.batch_size * math.floor(#data.train / self.batch_size), 4096),
        ['image_id'] = {},
        ['nbatches'] = math.floor(#data.train / self.batch_size)
    }

    if opt.gpuid >= 0 then
        self.batch_data.train.image_feat = self.batch_data.train.image_feat:cuda()
    end

    for i = 1, self.batch_size * self.batch_data.train.nbatches do
        self.batch_data.train.question[i] = data.train[i]['question']
        self.batch_data.train.answer[i] = data.train[i]['answer']
        self.batch_data.train.image_feat[i] = fc7[fc7_mapping[data.train[i]['image_id']]]
        self.batch_data.train.image_id[i] = data.train[i]['image_id']
    end

    if opt.gpuid >= 0 then
        self.batch_data.train.question = self.batch_data.train.question:cuda()
        self.batch_data.train.answer = self.batch_data.train.answer:cuda()
    end

    -- Load val into batches

    print('Loading val fc7 features from ' .. opt.val_fc7_file)
    local fc7 = torch.load(opt.val_fc7_file)
    local fc7_image_id = torch.load(opt.val_fc7_image_id_file)
    local fc7_mapping = {}
    for i, v in pairs(fc7_image_id) do
        fc7_mapping[v] = i
    end

    self.batch_data.val = {
        ['question'] = torch.ShortTensor(self.batch_size * math.floor(#data.val / self.batch_size), data.q_max_length),
        ['answer'] = torch.ShortTensor(self.batch_size * math.floor(#data.val / self.batch_size)),
        ['image_feat'] = torch.DoubleTensor(self.batch_size * math.floor(#data.val / self.batch_size), 4096),
        ['image_id'] = {},
        ['nbatches'] = math.floor(#data.val / self.batch_size)
    }

    if opt.gpuid >= 0 then
        self.batch_data.val.image_feat = self.batch_data.val.image_feat:cuda()
    end

    for i = 1, self.batch_size * self.batch_data.val.nbatches do
        self.batch_data.val.question[i] = data.val[i]['question']
        self.batch_data.val.answer[i] = data.val[i]['answer']
        self.batch_data.val.image_feat[i] = fc7[fc7_mapping[data.val[i]['image_id']]]
        self.batch_data.val.image_id[i] = data.val[i]['image_id']
    end

    if opt.gpuid >= 0 then
        self.batch_data.val.question = self.batch_data.val.question:cuda()
        self.batch_data.val.answer = self.batch_data.val.answer:cuda()
    end

    self.train_batch_idx = 1
    self.val_batch_idx = 1

    collectgarbage()
    return self

end

function DataLoader:next_batch(split)
    split = split or 'train'
    if split == 'train' then
        if self.train_batch_idx - 1 == self.batch_data.train.nbatches then self.train_batch_idx = 1 end
        local question = self.batch_data.train.question:narrow(1, (self.train_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local answer = self.batch_data.train.answer:narrow(1, (self.train_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image = self.batch_data.train.image_feat:narrow(1, (self.train_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image_id = {unpack(self.batch_data.train.image_id, (self.train_batch_idx - 1) * self.batch_size + 1, self.train_batch_idx * self.batch_size)}

        self.train_batch_idx = self.train_batch_idx + 1
        return question, answer, image, image_id
    else
        if self.val_batch_idx - 1 == self.batch_data.val.nbatches then self.val_batch_idx = 1 end
        local question = self.batch_data.val.question:narrow(1, (self.val_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local answer = self.batch_data.val.answer:narrow(1, (self.val_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image = self.batch_data.val.image_feat:narrow(1, (self.val_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image_id = {unpack(self.batch_data.val.image_id, (self.val_batch_idx - 1) * self.batch_size + 1, self.val_batch_idx * self.batch_size)}

        self.val_batch_idx = self.val_batch_idx + 1
        return question, answer, image, image_id
    end
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
    max_length = 0

    for i = 1, #train_q['questions'] do
        local count = 0
        if a_vocab_mapping[train_a['annotations'][i]['multiple_choice_answer']] then
            for token in word_iter(train_q['questions'][i]['question']) do
                if not unordered[token] then
                    unordered[token] = 1
                else
                    unordered[token] = unordered[token] + 1
                end
                count = count + 1
            end
            if count > max_length then max_length = count end
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
        if count > max_length then max_length = count end
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
        q_max_length = max_length
    }

    print('q max length: ' .. max_length)

    local idx = 1

    for i = 1, #train_q['questions'] do
        if a_vocab_mapping[train_a['annotations'][i]['multiple_choice_answer']] then
            local question = {}
            local wl = 0
            for token in word_iter(train_q['questions'][i]['question']) do
                wl = wl + 1
                question[wl] = q_vocab_mapping[token] or q_vocab_mapping["UNK"]
            end
            data.train[idx] = {
                image_id = train_a['annotations'][i]['image_id'],
                question = torch.ShortTensor(max_length):zero(),
                answer = a_vocab_mapping[train_a['annotations'][i]['multiple_choice_answer']] or a_vocab_mapping["UNK"]
            }
            for j = 1, wl do
                data.train[idx]['question'][max_length - wl + j] = question[j]
            end
            idx = idx + 1
        end
    end

    idx = 1

    for i = 1, #val_q['questions'] do
        local question = {}
        local wl = 0
        for token in word_iter(val_q['questions'][i]['question']) do
            wl = wl + 1
            question[wl] = q_vocab_mapping[token] or q_vocab_mapping["UNK"]
        end
        data.val[idx] = {
            image_id = val_a['annotations'][i]['image_id'],
            question = torch.ShortTensor(max_length):zero(),
            answer = a_vocab_mapping[val_a['annotations'][i]['multiple_choice_answer']] or a_vocab_mapping["UNK"]
        }
        for j = 1, wl do
            data.val[idx]['question'][max_length - wl + j] = question[j]
        end
        idx = idx + 1
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
