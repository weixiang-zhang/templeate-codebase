import torch
class Transform(object):
    def __init__(self, args):

        self.inverse_map = {}
        self.args = args

        self.method = self.args.inr_transform
    
    def tranform(self, data):
        if self.method == "min_max":
            _min = torch.min(data)
            _max = torch.max(data)
            self.inverse_map[self.method] = (_min, _max)
            normed = (data - _min) / (_max - _min)
            normed = self._encode_shift_scale(normed)
        else: 
            raise NotImplementedError

        return normed

    def inverse(self, normed):
        if self.method == "min_max":
            # scale → shift
            normed = self._decode_shift_scale(normed)
            _min, _max = self.inverse_map[self.method]
            r_data = normed  * (_max - _min) + _min

        else: 
            raise NotImplementedError
        
        return r_data
    
    def _encode_shift_scale(self, data):
        data = data +  self.args.trans_shift
        data = data * self.args.trans_scale
        return data
    
    def _decode_shift_scale(self, data):
        data = data / self.args.trans_scale
        data = data -  self.args.trans_shift
        return data

   

            