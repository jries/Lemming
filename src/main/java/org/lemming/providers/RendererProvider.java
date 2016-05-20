package org.lemming.providers;

import org.lemming.factories.RendererFactory;

public class RendererProvider extends AbstractProvider<RendererFactory> {

	public RendererProvider() {
		super(RendererFactory.class);
	}

	public static void main(String[] args) {
		final RendererProvider provider = new RendererProvider();
		System.out.println( provider.echo() );
	}

}
