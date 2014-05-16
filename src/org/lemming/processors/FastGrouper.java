package org.lemming.processors;

import java.util.Arrays;
import java.util.List;

import org.lemming.data.HashWorkspace;
import org.lemming.data.Localization;
import org.lemming.utils.SortWorkspace;
/**
 * groups localizations in adjacent frames together
 * 
 * 
 * @author ries
 *
 */
public class FastGrouper extends SISO<HashWorkspace,Localization> {

	@Override
	public void process(HashWorkspace ws) {
		float dX=1; 
		int dT=1;
		
		String[] memberList={"frame","x"};
		SortWorkspace sorter=new SortWorkspace(ws,memberList);
		Integer[] sInd=sorter.SortedIndices;
		List<Long> Frames=ws.getMember("frame");
		List<Double> x=ws.getMember("x");
		List<Double> y=ws.getMember("y");
		int NumberOfLocs=ws.getNumberOfRows();
		
		
		/* 
		 * initialize Group number with null
		 */
		Integer[] Group=new Integer[NumberOfLocs];
		Arrays.fill(Group, null);
		Group[sInd[0]]=0;
		int GroupInd=1;
		
		Integer[] k=new Integer[dT];
		Integer[] ksave =new Integer[dT];
		Arrays.fill(k, 1);
		/*
		 * put k to start of next frame
		 */
		int jn=0;
		while(jn<NumberOfLocs-1){
			for(int nT=1;nT<=dT;nT++){
				while(k[nT]<NumberOfLocs&&Frames.get(sInd[k[nT]])<Frames.get(sInd[jn])+nT){
					k[nT]++;
				}
			}
		
			/*
			 * move k until x(k) within dX
			 */
			for(int nT=1;nT<=dT;nT++){
				while(k[nT]<NumberOfLocs&&Frames.get(sInd[k[nT]])==Frames.get(sInd[jn])+nT&&x.get(sInd[k[nT]])<x.get(sInd[jn])-dX){
					k[nT]++;
				}
			}
			ksave=Arrays.copyOf(k, k.length);
			/*
			 * test while within range
			 */
			for(int nT=1;nT<=dT;nT++){
				while(k[nT]<NumberOfLocs&&Frames.get(sInd[k[nT]])==Frames.get(sInd[jn])+nT&&x.get(sInd[k[nT]])<x.get(sInd[jn])+dX){
					if(Math.abs(y.get(sInd[k[nT]])-y.get(sInd[jn]))<=dX){
						Group[sInd[k[nT]]]=Group[sInd[jn]];
						break;
					}
					k[nT]++;
				}
			}
			k=Arrays.copyOf(ksave, ksave.length);
			jn++;
			if(Group[sInd[jn]]==null){
				Group[sInd[jn]]=GroupInd;
				GroupInd++;
			}
		}
	}

}
