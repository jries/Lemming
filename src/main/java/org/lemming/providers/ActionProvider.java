package org.lemming.providers;

import org.lemming.factories.ActionFactory;


public class ActionProvider extends AbstractProvider<ActionFactory> {

	public ActionProvider() {
		super(ActionFactory.class);
	}

	public static void main(String[] args) {
		final ActionProvider provider = new ActionProvider();
		System.out.println( provider.echo() );
	}

}
