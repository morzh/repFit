# VideoFilter

Filter fitness videos by a list of criteria:
1) Video must have a person more than 4 sec with less than 5% gaps
2) Person joints prediction must be with more than 0.9 credibility in a middle from all skeleton joints
3) Video must have only one stable person tracking in one moment of time (with less than 5% crossing).



Future features:
1) split video frame in persons always are in different regions
2) add weights for filter by credibility