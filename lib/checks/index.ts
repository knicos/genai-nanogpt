import { execute as rope } from './rope';
import { execute as normRMS } from './normRMS';
import { execute as qkv } from './qkv';
import { execute as gelu } from './gelu';
import { execute as normRMSGrad } from './normRMSGrad';
import { execute as appendCache } from './appendCache';
import { execute as attentionMask } from './attentionMask';
import runCheck from './check';

const checks = {
    rope,
    qkv,
    gelu,
    normRMS,
    normRMSGrad,
    appendCache,
    attentionMask,
    runCheck,
};

export default checks;
