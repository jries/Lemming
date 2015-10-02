package org.lemming.modules;

import net.imglib2.type.numeric.RealType;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.MultiRunModule;
import java.util.concurrent.ConcurrentLinkedQueue;;

public abstract class Detector<T extends RealType<T>, F extends Frame<T>> extends MultiRunModule {
	
	private long start;

	//private volatile int counter= 0;
	
	private ConcurrentLinkedQueue<Integer> counterList = new ConcurrentLinkedQueue<>();

	public Detector() {
	}
	
	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
	}

	@SuppressWarnings("unchecked")
	@Override
	public Element process(Element data) {
		F frame = (F) data;
		if (frame == null)
			return null;

		if (frame.isLast()) { // make the poison pill
			cancel();
			FrameElements<T> res = detect(frame);
			res.setLast(true);
			//counter += res.getList().size();
			counterList.add(res.getList().size());
			return res;
		}
		FrameElements<T> res = detect(frame);
		//counter += res.getList().size();
		counterList.add(res.getList().size());
		return res;
	}
	
	public abstract FrameElements<T> detect(F frame);
		
	
	@Override
	protected void afterRun() {
		Integer cc=0;
		for (Integer i : counterList)
			cc+=i;
		System.out.println("Detector found "
				+ cc + " peaks in "
				+ (System.currentTimeMillis() - start) + "ms.");
	}

	@Override
	public boolean check() {
		return inputs.size()==1 && outputs.size()>=1;
	}

}
