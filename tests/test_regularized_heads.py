"""Train-only normalization, finite shrinkage and native linear-layer equivalence."""
import unittest

import torch
from torch import nn

from overfitted_underqualified.ml.regularized_heads import (
    fit_logistic, fit_shrinkage_lda, fold_standardization, weighted_standardize,
)


class RegularizedHeadsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.x=torch.tensor([[-2.,3.,1.],[-1.,3.,2.],[1.,3.,1.],[2.,3.,2.]])
        self.y=torch.tensor([0,0,1,1])
        self.p=torch.full((4,),.25,dtype=torch.float64)

    def test_standardization_is_train_weighted_and_handles_constant_feature(self):
        z,y,p,mean,scale=weighted_standardize(self.x,self.y,self.p,.001)
        torch.testing.assert_close(mean,torch.tensor([0.,3.,1.5],dtype=torch.float64))
        self.assertTrue(torch.isfinite(z).all())
        self.assertEqual(float(scale[1]),.001)
        self.assertTrue(torch.equal(z[:,1],torch.zeros(4,dtype=torch.float64)))

    def test_folding_preserves_unseen_logits_not_just_training_logits(self):
        _,_,_,mean,scale=weighted_standardize(self.x,self.y,self.p,.001)
        head=nn.Linear(3,2,dtype=torch.float64)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[.2,0.,-.3],[-.1,0.,.4]],dtype=torch.float64))
        folded=fold_standardization(head,mean,scale)
        unseen=torch.tensor([[8.,5.,-.4],[-7.,2.,.3]],dtype=torch.float64)
        torch.testing.assert_close(head((unseen-mean)/scale),folded(unseen),atol=1e-10,rtol=1e-10)
        self.assertIsInstance(folded,nn.Linear)

    def test_both_solvers_are_finite_and_keep_class_order(self):
        for fit,kwargs in ((fit_logistic,dict(l2=1.,max_iter=100)),(fit_shrinkage_lda,dict(shrinkage=.9))):
            head,metadata=fit(self.x,self.y,self.p,variance_floor=.001,**kwargs)
            self.assertEqual(head(self.x).argmax(1).tolist(),self.y.tolist())
            self.assertTrue(torch.isfinite(head.weight).all())
            self.assertLess(metadata['fold_error_float64'],1e-8)

    def test_duplicate_rows_with_split_mass_do_not_create_more_evidence(self):
        a,_=fit_logistic(self.x,self.y,self.p,l2=1.,max_iter=100,variance_floor=.001)
        b,_=fit_logistic(self.x.repeat_interleave(2,0),self.y.repeat_interleave(2),
            self.p.repeat_interleave(2)/2,l2=1.,max_iter=100,variance_floor=.001)
        torch.testing.assert_close(a.weight,b.weight,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(a.bias,b.bias,atol=1e-6,rtol=1e-6)

    def test_invalid_or_single_class_banks_fail(self):
        with self.assertRaises(ValueError):weighted_standardize(self.x,torch.zeros(4,dtype=torch.long),self.p,.001)
        with self.assertRaises(ValueError):weighted_standardize(self.x,self.y,self.p*2,.001)
        with self.assertRaises(ValueError):fit_logistic(self.x,self.y,self.p,l2=0,max_iter=5,variance_floor=.001)


if __name__=='__main__':unittest.main()
