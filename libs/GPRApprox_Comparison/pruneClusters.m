function ci = pruneClusters(ci, cSize)

n = length(unique(ci));
for i=0:n-1
    ids = find(ci==i);
    excess = length(ids)-cSize;
    if excess > 0
        idsPerm = randperm(length(ids));
        ci(ids(idsPerm(1:excess)))=-1; % Negative ids are ignored.
    end
end
